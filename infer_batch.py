import os
import time
import librosa
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from kimia_infer.api.kimia import KimiAudio

warnings.filterwarnings("ignore")

sampling_params = {
    "audio_temperature": 0.0,
    "audio_top_k": 1,
    "text_temperature": 0.0,
    "text_top_k": 1,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

SCORE_CODE_MAP = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10}

def load_text(file:str):
    text_dict = {}
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split("\t")
        if len(line) < 2: continue
        key, text = line[0], line[1]
        text_dict[key] = text
    print(f'Total samples for inference: {len(text_dict)}')
    return text_dict

def load_audio(file:str):
    audio_dict = {}
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split("\t", maxsplit=1)
        if len(parts) < 2: continue
        key, audio_path = parts[0], parts[1]
        audio_dict[key] = audio_path
    return audio_dict

def infer_statistic(infer_time_list: list, batch_sizes: list, total_duration: float, statistic_fo:str):
    if not infer_time_list:
        print("No inference records collected, skip statistics.")
        return

    n_batches = len(infer_time_list)
    n_samples = int(sum(batch_sizes))
    total_infer_time = sum(infer_time_list)  # ms

    qps = round(n_samples / (total_infer_time / 1000), 4) if total_infer_time > 0 else 0
    avg_infer_time = round(total_infer_time / n_samples, 4) if n_samples > 0 else 0

    percentiles = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
    infer_time_percentiles = np.round(np.percentile(infer_time_list, percentiles), 4)

    df = pd.DataFrame({
        'Percentile': percentiles,
        'Inference Time (ms)': infer_time_percentiles
    })
    print("\nInference Statistics:")
    print(f"Total Samples: {n_samples}")
    print(f"Total Batches: {n_batches}")
    print(f"Total Inference Time (ms): {total_infer_time:.2f}")
    print(f"Total Audio Duration (ms): {total_duration:.2f}")
    print(f"QPS: {qps} it/s")
    print(f"Average Inference Time per Sample (ms): {avg_infer_time}")
    print("\nPercentile Statistics:")
    print(df.to_string(index=False))
    df.to_csv(statistic_fo, index=True, encoding='utf-8', sep='\t')

def batch_inference(model:KimiAudio, batch_data: list, max_new_tokens:int=1) -> tuple[list, list]:
    # batch_data: list of (input_text, input_audio) tuples
    chats_batch = [
        (
            [{"role": "user", "message_type": "text", "content": input_text}]
            if input_audio is None
            else [
                {"role": "user", "message_type": "text", "content": input_text},
                {"role": "user", "message_type": "audio", "content": input_audio},
            ]
        )
        for input_text, input_audio in batch_data
    ]
    
    # model.generate should now support batch
    texts, probs = model.generate(
        chats_batch,
        **sampling_params,
        output_type="text",
        max_new_tokens=max_new_tokens,
    )
    
    # texts is list of string sentences (detokenized)
    return texts, probs

def main_single_dataset_batch(
        model_path:str,
        infer_file:str,
        audio_file:str=None,
        batch_size:int=8,
        gpu_id:int=0
    ):
    device = f'cuda:{gpu_id}'
    model = KimiAudio(model_path=model_path, load_detokenizer=False, device=device)

    infer_text_content = '根据音频和评测文本，评测句子整体发音准确性，评分按顺序分为a,b,c,d到k共11档，a档表示0分，k档表示10分，每个档跨度是1分，最后输出档次。评测文本：{}'

    if audio_file is None:
        audio_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/api_data/next/tal-k12/wavpath_merged'

    text_dict = load_text(infer_file)
    audio_dict = load_audio(audio_file)
    shared_key = set(text_dict.keys()) & set(audio_dict.keys())
    # Sort keys to ensure deterministic order
    sorted_keys = sorted(list(shared_key))
    
    print(f"Found {len(sorted_keys)} samples with both text and audio.")

    fo_path = os.path.join(model_path, 'infer_res_batch', os.path.basename(infer_file))
    statistic_fo = fo_path + '.statistic.tsv'
    if not os.path.exists(os.path.dirname(fo_path)):
        os.makedirs(os.path.dirname(fo_path), exist_ok=True)
    fo = open(fo_path, "w", encoding="utf-8")

    total_duration = 0 # ms
    infer_time_list, batch_sizes = [], []
    
    # Prepare all data items
    all_data = []
    print("Pre-loading data durations...")
    for key in tqdm(sorted_keys):
        ref_text = text_dict[key]
        infer_audio_content = audio_dict.get(key, None)
        if infer_audio_content is None: continue
        
        input_text = infer_text_content.format(ref_text)
        try:
            duration = (librosa.get_duration(filename=infer_audio_content, sr=16000)) * 1000  # ms
        except Exception as e:
            print(f"Error loading audio {infer_audio_content}: {e}")
            continue

        all_data.append({
            'key': key,
            'input_text': input_text,
            'input_audio': infer_audio_content,
            'duration': duration
        })

    print("Sorting data by duration...")
    all_data.sort(key=lambda x: x['duration'])
    if not all_data:
        print("No valid samples found, exiting.")
        fo.close()
        return
    print(f"Max duration: {max([item['duration'] for item in all_data])} ms")
    # Warmup
    print("Warming up model with 10 samples...")
    warmup_samples = all_data[:10]
    for i in range(0, len(warmup_samples), batch_size):
        batch_items = warmup_samples[i : i + batch_size]
        batch_inputs = [(item['input_text'], item['input_audio']) for item in batch_items]
        try:
            batch_inference(model, batch_inputs, max_new_tokens=1)
        except Exception as e:
            print(f"Warmup error: {e}")

    # Batch Process
    print(f'Starting batch inference with Batch Size = {batch_size}')
    for i in tqdm(range(0, len(all_data), batch_size), desc="Batch Inference"):
        batch_items = all_data[i : i + batch_size]
        batch_inputs = [(item['input_text'], item['input_audio']) for item in batch_items]
        
        start_time = time.time()
        texts, text_probs = batch_inference(model, batch_inputs, max_new_tokens=1)

        end_time = time.time()
        batch_infer_time = (end_time - start_time) * 1000 # ms
        infer_time_list.append(batch_infer_time)
        batch_sizes.append(len(batch_items))
        for j, item in enumerate(batch_items):
            duration = item['duration']
            total_duration += duration

            text = texts[j]
            probs = text_probs[j]
            score = text.strip()
            score_code = str(SCORE_CODE_MAP.get(score, -1))
            
            fo.write(f"{item['key']}\t{score_code}\t{probs}\n")
        
        fo.flush()

    fo.close()
    infer_statistic(infer_time_list, batch_sizes, total_duration, statistic_fo)

def main_asr():
    infer_data='/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/audio_detect/test/noise_asr_testdataset.csv'
    model_path = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/Base_Model/Kimi-PA-Base-v3/CPT_STAGE1_MODEL/infer_cpkt9k'
    # model_path = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/Kimi-Audio-7B-Instruct'

    infer_text_content = "Please transcribe the following audio:"
    model = KimiAudio(model_path=model_path, load_detokenizer=False, device=f'cuda:0')
    data = pd.read_csv(infer_data, sep='\t', usecols=['wavname', 'text', 'wavpath'])
    data = data[data['wavname'].isin(['17676960006241458168243156193280', '17654562927241448774227352285184'])]
    batch_size = 1
    for i in tqdm(range(0, len(data), batch_size), desc="ASR Batch Inference"):
        batch_data = data.iloc[i : i + batch_size]
        batch_inputs = []
        for _, row in batch_data.iterrows():
            input_text = infer_text_content
            input_audio = row['wavpath']
            batch_inputs.append((input_text, input_audio))
        
        texts, _ = batch_inference(model, batch_inputs, max_new_tokens=100)
        for j, (_, row) in enumerate(batch_data.iterrows()):
            print(f"Audio: {row['wavname']}, Reference: {row['text']}, ASR Output: {texts[j]}")
            # print(f'{row["wavname"]}\t{row["text"]}\t{texts[j]}')


if __name__ == "__main__":
    model_path = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v3.3/infer_model'
    # model_path = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/Base_Model/Kimi-PA-Base-v3/CPT_STAGE1_MODEL/infer_cpkt9k'

    infer_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/api_data/next/tal-k12/test/label_sent_score'
    # infer_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/api_data/next/tal-k12/test/label_snt_score_batch2'
    # infer_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/audio_detect/test/label_snt_score_ad'
    # infer_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/api_data/next/tal-k12/test/label_snt_score_merged'


    # You can change batch_size here
    main_single_dataset_batch(model_path, infer_file, batch_size=5)
    # main_single_dataset_batch(model_path, infer_file, batch_size=8, audio_file='/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/audio_detect/test/wavpath')

    # main_asr()

