import os
import sys
import json
import time
import librosa
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from kimia_infer.api.kimia import KimiAudio
import torch

warnings.filterwarnings("ignore")

sampling_params = {
    "audio_temperature": 0.0, # Greedy
    "audio_top_k": 1,
    "text_temperature": 0.0,
    "text_top_k": 1,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

SCORE_CODE_MAP = {
        '0': '正常',  
        '1': '噪声',
        '2': '不相关中文',
        '3': '不相关英文',
        '4': '无意义语音',
        '5': '音量小',
        '6': '开头发音不完整',
        '7': '空音频'
    }

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

def infer_statistic(rtf_list: list, infer_time_list: list, total_duration: float, statistic_fo:str):
    if not rtf_list: return
    n_samples = len(rtf_list)
    total_infer_time = sum(infer_time_list)  # ms

    overall_rtf = round(total_infer_time / total_duration, 4) if total_duration > 0 else 0
    qps = round(n_samples / (total_infer_time / 1000), 4) if total_infer_time > 0 else 0
    avg_infer_time = round(total_infer_time / n_samples, 4)

    percentiles = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
    infer_time_percentiles = np.round(np.percentile(infer_time_list, percentiles), 4)
    rtf_percentiles = np.round(np.percentile(rtf_list, percentiles), 4)

    df = pd.DataFrame({
        'Percentile': percentiles,
        'RTF': rtf_percentiles,
        'Inference Time (ms)': infer_time_percentiles
    })
    print("\nInference Statistics:")
    print(f"Total Samples: {n_samples}")
    print(f"Total Inference Time (ms): {total_infer_time:.2f}")
    print(f"Total Audio Duration (ms): {total_duration:.2f}")
    print(f"Overall RTF: {overall_rtf}")
    print(f"QPS: {qps} it/s")
    print(f"Average Inference Time per Sample (ms): {avg_infer_time}")
    print("\nPercentile Statistics:")
    print(df.to_string(index=False))
    df.to_csv(statistic_fo, index=True, encoding='utf-8', sep='\t')

def batch_inference(model:KimiAudio, batch_data: list, max_new_tokens:int=1) -> tuple[list, list]:
    # batch_data: list of (input_text, input_audio) tuples
    chats_batch = []
    for input_text, input_audio in batch_data:
        if input_audio is None:
            messages = [{"role": "user", "message_type": "text", "content": input_text}]
        else:
            messages = [
                {"role": "user", "message_type": "text", "content": input_text},
                {"role": "user", "message_type": "audio", "content": input_audio},
            ]
        chats_batch.append(messages)
    
    # model.generate should now support batch
    wavs, texts, probs = model.generate(chats_batch, **sampling_params, output_type="text", max_new_tokens=max_new_tokens)
    
    # texts is list of string sentences (detokenized)
    return texts, probs

def main_single_dataset_batch(
        model_path:str,
        infer_file:str,
        audio_file:str,
        fo_path:str,
        batch_size:int=8,
        gpu_id:int=0
    ):
    device = f'cuda:{gpu_id}'
    model = KimiAudio(model_path=model_path, load_detokenizer=False, device=device)

    infer_text_content = '检测音频类型，包含正常、噪声、不相关中文、不相关英文、无意义语音、音量小、开头发音不完整、空音频八类，分别对应0、1、2、3、4、5、6、7，根据参考文本做出判断。参考文本：{}'

    text_dict = load_text(infer_file)
    audio_dict = load_audio(audio_file)
    shared_key = set(text_dict.keys()) & set(audio_dict.keys())
    # Sort keys to ensure deterministic order
    sorted_keys = sorted(list(shared_key))
    
    print(f"Found {len(sorted_keys)} samples with both text and audio.")

    statistic_fo = fo_path + '.statistic.tsv'
    if not os.path.exists(os.path.dirname(fo_path)):
        os.makedirs(os.path.dirname(fo_path), exist_ok=True)
    fo = open(fo_path, "w", encoding="utf-8")

    total_duration = 0 # ms
    rtf_list, infer_time_list = [], []
    
    # Prepare all data items
    all_data = []
    print("Pre-loading data durations...")
    for key in tqdm(sorted_keys):
        ref_text = text_dict[key]
        infer_audio_content = audio_dict.get(key, None)
        if infer_audio_content is None: continue
        
        input_text = infer_text_content.format(ref_text)
        # Note: librosa.get_duration requires file IO enabled if passing filename. 
        # For speed, we might want to skip duration check here if not strictly needed before inference.
        # But we need it for statistics.
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

    # Sort data by duration to minimize padding overhead
    # This significantly improves QPS by grouping similar length audios together
    print("Sorting data by duration...")
    # all_data.sort(key=lambda x: x['duration'])
    print(f'Max duration: {max([item["duration"] for item in all_data])} ms')
    # sys.exit(0)
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
        # Batch inference call
        try:
            texts, text_probs = batch_inference(model, batch_inputs, max_new_tokens=1)
        except Exception as e:
            print(f"Error in batch inference: {e}")
            continue

        end_time = time.time()
        
        batch_infer_time = (end_time - start_time) * 1000 # ms
        avg_infer_time = batch_infer_time / len(batch_items)

        for j, item in enumerate(batch_items):
            duration = item['duration']
            total_duration += duration
            infer_time_list.append(avg_infer_time) 
            if duration > 0:
                rtf_list.append(avg_infer_time / duration)
            
            text = texts[j]
            probs = text_probs[j]
            score = text.strip()
            score_code = str(SCORE_CODE_MAP.get(score, -1))
            
            fo.write(f"{item['key']}\t{score_code}\t{probs}\n")
        
        fo.flush()

    fo.close()
    infer_statistic(rtf_list, infer_time_list, total_duration, statistic_fo)

if __name__ == "__main__":
    model_path = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v3.3/infer_model'
    infer_file = '/mnt/pfs_l2/jieti_team/SFT/speech/fangdongyan/fdy_10.19.36.121/00-code_backup/tal_kh_evl_tools/data_pipeline/files/260206/audio_type.csv'
    audio_file = '/mnt/pfs_l2/jieti_team/SFT/speech/fangdongyan/fdy_10.19.36.121/00-code_backup/tal_kh_evl_tools/data_pipeline/files/260206/audio_path'
    fo_path = '/mnt/pfs_l2/jieti_team/SFT/speech/fangdongyan/fdy_10.19.36.121/00-code_backup/tal_kh_evl_tools/data_pipeline/files/260206/audio_type_file.ad.compare_new'
    
    # You can change batch_size here

    main_single_dataset_batch(
        model_path, 
        infer_file, 
        audio_file, 
        fo_path,
        batch_size=8)

