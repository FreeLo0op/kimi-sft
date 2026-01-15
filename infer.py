import os
import sys
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

def inference(model:KimiAudio, input_text:str, input_audio:str) -> tuple[str, list]:
    if input_audio == None:
        messages = [
            {"role": "user", "message_type": "text", "content": input_text},
        ]
    else:
        messages = [
            {"role": "user", "message_type": "text", "content": input_text},
            {"role": "user", "message_type": "audio", "content": input_audio},
        ]
    _, text, text_probs = model.generate(messages, **sampling_params, output_type="text", max_new_tokens=1)
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
        infer_file:str
    ):
    model = KimiAudio(model_path=model_path, load_detokenizer=False, device=f'cuda:0')

    # infer_text_content = '你是一个KET打分考官，你需要根据考试问题和学生作答音频，对学生的回答进行发音评测。1、根据作答音频识别出文本；2、根据考试问题和识别结果，匹配出真正的作答内容；3、最后根据作答内容和音频，评测学生的句子发音准确性，评分标准为0-10分。[Pronunciation assessment for KET exam] 考题：{}'
    # infer_text_content = '你是一个英文语音评测助手，请根据音频和参考文本，评测句子整体发音准确性，并直接输出分数，评分标准为0-10分。[TASK:Sentence-level pronunciation assessment (accuracy) for non-native English-learning children(k12)] ,参考文本:{}' # v2.3
    # infer_text_content = '评测句子发音准确性，评分为a,b,c到k共11档。评测文本：{}'
    # infer_text_content = '根据音频和评测文本，评测句子整体发音准确性，评分从低到高分为a,b,c到u共21档。评测文本：{}'
    # infer_text_content = '根据音频和评测文本，评测句子整体发音准确性，评分按顺序分为a,b,c,d到u共21>档，a档表示0分，u档表示10分，每个档跨度是0.5分，最后输出档次。评测文本：{}'
    # infer_text_content = '你是一个英文语音评测助手，请根据音频和参考文本和音素，一步步完成音素、单词、句子三个层次的评测。首先，给出音素（准确度）评测结果，然后，给出单词（准确度）评测结果，最后，给出句子（准确度、流利度）评测结果。其中，音素评分标准为0-2分，单词评分标准为0-3分，句子准确度评分标准为0-10分，句子流利度评分标准为0-3分。English full pronunciation assessment：完成音素、单词、句子三个层次的评测。参考文本：{}。'
    infer_text_content = '根据音频和评测文本，评测句子整体发音准确性，评分按顺序分为a,b,c,d到k共11档，a档表示0分，k档表示10分，每个档跨度是1分，最后输出档次。评测文本：{}' # v3.4
    # infer_text_content = '根据音频和评测文本，评测句子整体发音准确性，评分按顺序从低到高分为a,b,c,d到k共11档。{}' # v3.5

    audio_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/wavpath_merged'

    text_dict = load_text(infer_file)
    audio_dict = load_audio(audio_file)
    shared_key = set(text_dict.keys()) & set(audio_dict.keys())
    text_dict = {k: text_dict[k] for k in shared_key}
    audio_dict = {k: audio_dict[k] for k in shared_key}
    
    fo_path = os.path.join(model_path, 'infer_res', os.path.basename(infer_file))
    statistic_fo = fo_path + '.statistic.tsv'
    if not os.path.exists(os.path.dirname(fo_path)):
        os.makedirs(os.path.dirname(fo_path), exist_ok=True)
    fo = open(fo_path, "w", encoding="utf-8")

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

def main_abnormal_dataset(
        model_path:str,
        infer_type:str='score'    
    ):
    data_input = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/audio_detect/test/label_only_abnormal.csv'
    # audio_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/abnormal_data/wavpath'
    
    model = KimiAudio(model_path=model_path, load_detokenizer=False, device=f'cuda:0')
    
    if infer_type == 'score':
        # infer_text_content = '你是一个英文语音评测助手，请根据音频和参考文本，评测句子整体发音准确性，并直接输出分数，评分标准为0-10分。[TASK:Sentence-level pronunciation assessment (accuracy) for non-native English-learning children(k12)] ,参考文本:{}'
        # infer_text_content = '评测句子发音准确性，评分为a,b,c到k共11档。评测文本：{}'
        # infer_text_content = '根据音频和评测文本，评测句子整体发音准确性，评分按顺序分为a,b,c,d到k共11档，a档表示0分，k档表示10分，每个档跨度是1分，最后输出档次。评测文本：{}' # v3.4
        infer_text_content = '根据音频和评测文本，评测句子整体发音准确性，评分按顺序从低到高分为a,b,c,d到k共11档。{}' # v3.5
        infer_fo = os.path.join(model_path, 'infer_res', 'abnormal_dataset.tsv')
        statistic_fo = infer_fo + '.statistic.tsv'
    elif infer_type == 'audio_type':
        infer_text_content = '检测音频类型，包含正常、噪声、不相关中文、不相关英文、无意义语音、音量小、开头发音不完整、空音频八类，分别对应0、1、2、3、4、5、6、7，根据参考文本做出判断。参考文本：{}'
        infer_fo = os.path.join(model_path, 'infer_res', 'abnormal_dataset_audio_type.tsv')
    else:
        raise ValueError(f"Unsupported infer_type: {infer_type}") 

    if not os.path.exists(os.path.dirname(infer_fo)):
        os.makedirs(os.path.dirname(infer_fo), exist_ok=True)

    # audio_dict = load_audio(audio_file)
    data = pd.read_csv(data_input, sep='\t')
    fo = open(infer_fo, "w", encoding="utf-8")
    total_duration = 0 # ms
    rtf_list, infer_time_list = [], []
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Inference", disable=False):
        key = row['wavname']
        ref_text = row['text']
        # infer_audio_content = audio_dict.get(key, None)
        infer_audio_content = row['wavpath']
        if infer_audio_content is None:
            print(f"Warning: Audio file for key {key} not found. Skipping.")
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

        score = text.strip()
        if infer_type == 'audio_type':
            score = CODE_MAP.get(score, '未知类别')
        fo.write(f'{key}\t{score}\t{text_probs}\n')
        fo.flush()
    fo.close()
    if infer_type == 'score':
        infer_statistic(rtf_list, infer_time_list, total_duration, statistic_fo)

def main_single_data():
    model_path = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v2.7/model_infer'
    model = KimiAudio(model_path=model_path, load_detokenizer=False, device=f'cuda:0')

    infer_text_content = "你是一个智慧助手，检测下文是否存在语法错误，如果存在请指出错在哪里：Mike is my friend. He go to school yesterday."

    infer_audio_content = None

    text, text_probs = inference(
        model=model,
        input_text=infer_text_content,
        input_audio=infer_audio_content,
    )
    print(text)
    
if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # # infer_prompt_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/multi_task/sft/test/tal-k12_sent_pa_accuracy_nocot-v2_test.json'
    # model_path = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v3.1/model_infer_ephch1'
    # main(
    #     infer_prompt=infer_prompt_file,
    #     gpu_id=gpu_id,
    #     model_path=model_path
    # )
    model_path = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v3.3/infer_model'

    files = [
        '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_snt_score_batch1',
        # '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_snt_score_batch2',
        # '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_snt_score_batch4',
        # '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_sent_score'
    ]
    # for file in files:
        # main_single_dataset(model_path, file)

    file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_snt_score_merged'
    # file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_sent_score'
    main_single_dataset(model_path, file)

    # main_abnormal_dataset(model_path, 'audio_type')
    # main_single_data()