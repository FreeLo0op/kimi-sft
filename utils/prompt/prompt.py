import os
import json
from collections import defaultdict

def load_wavpath(wavpth:str) -> dict:
    wav_dict = dict()
    with open(wavpth, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            audio_key, audio_path = line.strip().split('\t', maxsplit=1)
            wav_dict[audio_key] = audio_path
    return wav_dict

def phoneme_pa_prompt_creator(wavptah:str, label:str, output_file:str):
    wav_dict = load_wavpath(wavptah)
    label_dict = defaultdict(list)

    with open(label, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, words, ipas, scores = line.strip().split('\t', maxsplit=3)
            words = words.split(' ')
            ipas = ipas.split(',')
            scores = scores.split(',')

            # ref_text = ' '.join(words)
            ref_text = []
            for w, p in zip(words, ipas):
                ref_text.append(f"{w}[{p}]")
            ref_text = ' '.join(ref_text)

            pred_content = []
            for w, p, s in zip(words, ipas, scores):
                pred_content.append(f"{w} [{p}]({s})")
            pred_text = '|'.join(pred_content)
            label_dict[key] = [ref_text, pred_text]
    
    prompts = []
    for key in label_dict:
        audio = wav_dict.get(key, None)
        if audio is None:
            continue
        ref_text, pred_text = label_dict[key]
        data = {
            "task_type": "understanding",
            "conversation":[
                {
                    "role": "user",
                    "message_type": "text",
                    # "content": f"评测文本:{ref_text}，请根据评测文本对下面的音频进行发音评分，并输出每个词的音素及其对应的发音得分。"
                    "content": f"评测文本:{ref_text}，请根据评测文本对下面的音频进行发音评分，并输出每个词的音素及其对应的发音得分。"
                },
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": audio
                },
                {
                    "role": "assistant",
                    "message_type": "text",
                    "content": pred_text
                }
            ]
        }
        prompts.append(data)
    with open(output_file, 'w', encoding='utf-8') as fo:
        for item in prompts:
            fo.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    # 训练集
    wavpath = "/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/wavpath"
    label = "/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/train/label_phones_ipa"
    output_file = "/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/sft/train/phoneme_pa_ipa_v2_train.jsonl"

    # 测试集
    wavpath = "/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/wavpath"
    label = "/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_phones_ipa"
    output_file = "/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/sft/test/phoneme_pa_ipa_v2_test.jsonl"
    phoneme_pa_prompt_creator(wavpath, label, output_file)

    


