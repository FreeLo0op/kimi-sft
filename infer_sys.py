import os
import sys
import ast
import json
import soundfile as sf
from tqdm import tqdm
import time
import librosa
import warnings
import numpy as np
import pandas as pd
from kimia_infer.api.kimia import KimiAudio
warnings.filterwarnings("ignore")

sampling_params = {
    "audio_temperature": 0.8,
    "audio_top_k": 1,
    "text_temperature": 0.0,
    "text_top_k": 1,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

CODE_MAP = {
        '0': '正常',  
        '1': '噪声',
        '2': '不相关中文',
        '3': '不相关英文',
        '4': '无意义语音',
        '5': '音量小',
        '6': '开头发音不完整',
        '7': '空音频'
    }

SCORE_CODE_MAP = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10}

def prompt_loader(prompt_path:str, if_convert:bool=True) -> list:
    infer_messages = []
    if prompt_path.endswith('.jsonl'):
        if_convert = False
    elif prompt_path.endswith('.json'):
        if_convert = True
    with open(prompt_path, "r", encoding="utf-8") as f:
        if if_convert:
            f = json.load(f)
            for item in f:
                single_messages = []
                messages = item["messages"]
                system_content = messages[0]["content"]
                user_content = messages[1]["content"]
                user_content = user_content.replace("<audio>", "").strip()
                infer_content = f"{system_content}{user_content}"

                label = item["messages"][2]["content"]
                # label = -1
                audio = item["audios"][0]
                # if '17000481161621174432450764361728' not in  audio:
                    # continue

                single_messages.append({"role": "user", "message_type": "text", "content": infer_content})
                single_messages.append({"role": "user", "message_type": "audio", "content": audio})
                # single_messages.append({"role": "user", "message_type": "audio-text", "content": [audio, infer_content]})
                single_messages.append({"role": "assistant_gt", "message_type": "text", "content": label})
                infer_messages.append(single_messages)
        else:
            for line in f:
                item = json.loads(line)
                single_messages = item['conversation']
                infer_messages.append(single_messages)
    return infer_messages

def inference(model:KimiAudio, input_text:str, input_audio:str, max_new_tokens:int=1) -> tuple[str, list]:
    if input_audio == None:
        messages = [
            {"role": "user", "message_type": "text", "content": input_text},
        ]
    else:
        messages = [
            {"role": "user", "message_type": "text", "content": input_text},
            {"role": "user", "message_type": "audio", "content": input_audio},
        ]
    _, text, text_probs = model.generate(messages, **sampling_params, output_type="text", max_new_tokens=max_new_tokens)
    text = ''.join(text)
    return text, text_probs

def main(
        infer_prompt:str,
        gpu_id:str,
        model_path:str="/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/Kimi_Pa_V1.1_hf_for_inference",
    ):
    infer_messages = prompt_loader(infer_prompt, if_convert=False)
    model = KimiAudio(model_path=model_path, load_detokenizer=False, device=f'cuda:{gpu_id}')
    output_path = os.path.join(model_path, 'infer_res', os.path.basename(infer_prompt))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fo = open(output_path, "w", encoding="utf-8")

    for i in tqdm(range(len(infer_messages)), desc="Inference", disable=False):
        messages = infer_messages[i][:-1]
        label = infer_messages[i][-1]["content"]
        audio = infer_messages[i][1]["content"]
        # print(messages)
        _, text, text_probs = model.generate(messages, **sampling_params, output_type="text")
        text = ''.join(text)
        # wav, text = model.generate(messages, **sampling_params, output_type="both")

        infer_res = {
            "prompt": messages[0]["content"],
            "audio": audio,
            "label": label,
            "predict": text,
        }
        fo.write(json.dumps(infer_res, ensure_ascii=False) + "\n")
        fo.flush()
    fo.close()
    
def load_text(file:str):
    text_dict = {}
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split("\t")
        key, text = line[0], line[1]
        if key in text_dict:
            print(f"Warning: Duplicate key found: {key}")
        text_dict[key] = text
    print(f'Total samples for inference: {len(text_dict)}')
    return text_dict

def load_audio(file:str):
    audio_dict = {}
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        key, audio_path = line.strip().split("\t", maxsplit=1)
        audio_dict[key] = audio_path
    return audio_dict

def infer_statistic(rtf_list: list, infer_time_list: list, total_duration: float, statistic_fo:str):
    
    assert len(rtf_list) == len(infer_time_list)
    n_samples = len(rtf_list)
    total_infer_time = sum(infer_time_list)  # ms

    overall_rtf = round(total_infer_time / total_duration, 4)
    qps = round(n_samples / (total_infer_time / 1000), 4)
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

def main_single_dataset(
        model_path:str,
        infer_file:str,
        audio_file:str,
        output_file:str
    ):
    model = KimiAudio(model_path=model_path, load_detokenizer=False, device=f'cuda:0')


    infer_text_content = '根据音频和评测文本，评测句子整体发音准确性，评分按顺序分为a,b,c,d到k共11档，a档表示0分，k档表示10分，每个档跨度是1分，最后输出档次。评测文本：{}' # v3.4 v3.3

    text_dict = load_text(infer_file)
    audio_dict = load_audio(audio_file)
    shared_key = set(text_dict.keys()) & set(audio_dict.keys())
    text_dict = {k: text_dict[k] for k in shared_key}
    audio_dict = {k: audio_dict[k] for k in shared_key}
    
    statistic_fo = output_file + '.statistic.tsv'
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fo = open(output_file, "w", encoding="utf-8")

    total_duration = 0 # ms
    count = 0
    rtf_list, infer_time_list = [], []
    for key in tqdm(text_dict, total=len(text_dict), desc="Inference", disable=False):
        # try:
        ref_text = text_dict[key]
        infer_audio_content = audio_dict.get(key, None)
        if infer_audio_content is None:
            continue
        input_text = infer_text_content.format(ref_text)
        duration = (librosa.get_duration(filename=infer_audio_content, sr=16000)) * 1000  # ms
        start_time = time.time()
        text, text_probs = inference(model, input_text, infer_audio_content)
        end_time = time.time()
        
        infer_time = (end_time - start_time) * 1000  # ms
        infer_time_list.append(infer_time)
        total_duration += duration
        if duration > 0:
            rtf_list.append(infer_time / duration)
            # print(f'{key}\t{infer_time}\t{duration}')

        score = text.strip()
        score = str(SCORE_CODE_MAP.get(score, -1))
            # fo.write(f'{key}\t{score}\t{text_probs}\n')
        # except Exception as e:
        #     print(f"Error processing {key}: {e}")
        #     score = 0
        fo.write(f'{key}\t{score}\t{text_probs}\n')
        fo.flush()
        # count += 1
        # if count == 100:break
    fo.close()

    infer_statistic(rtf_list, infer_time_list, total_duration, statistic_fo)    

if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    model_path = sys.argv[1]
    text_file = sys.argv[2]
    audio_file = sys.argv[3]
    output_file = sys.argv[4]

    main_single_dataset(model_path, text_file, audio_file, output_file)