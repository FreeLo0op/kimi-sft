import os
import sys
import json
from tqdm import tqdm
import time
import librosa
import warnings
import pandas as pd
from kimia_infer.api.kimia import KimiAudio
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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
    _, text, text_probs = model.generate(messages, **sampling_params, output_type="text", max_new_tokens=512)
    text = ''.join(text)
    return text, text_probs

def main_abnormal_dataset(
        model_path:str,
        data_input:str,
        infer_fo:str
    ):
    # data_input = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/audio_detect/test/label_only_abnormal.csv'    
    model = KimiAudio(model_path=model_path, load_detokenizer=False, device=f'cuda:0')
    
    infer_text_content = '检测音频类型，包含正常、噪声、不相关中文、不相关英文、无意义语音、音量小、开头发音不完整、空音频八类，分别对应0、1、2、3、4、5、6、7，根据参考文本做出判断。参考文本：{}'

    if not os.path.exists(os.path.dirname(infer_fo)):
        os.makedirs(os.path.dirname(infer_fo), exist_ok=True)

    data = pd.read_csv(data_input, sep='\t', header=None, names=["wavname", "text", "audio_type", "wavpath"])
    fo = open(infer_fo, "w", encoding="utf-8")

    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Inference", disable=False):
        key = row['wavname']
        ref_text = row['text']
        infer_audio_content = row['wavpath']
        
        # 检查wavpath是否为空（NaN、None或非字符串类型）
        if pd.isna(infer_audio_content) or infer_audio_content is None:
            print(f"Warning: Audio file path is empty for key {key}. Skipping.")
            continue
        
        # 确保是字符串类型
        if not isinstance(infer_audio_content, str):
            print(f"Warning: Audio file path is not a string for key {key}: {type(infer_audio_content)}. Skipping.")
            continue
        
        # 检查音频文件是否存在
        if not os.path.exists(infer_audio_content):
            print(f"Warning: Audio file does not exist for key {key}: {infer_audio_content}. Skipping.")
            continue
        input_text = infer_text_content.format(ref_text)
        
        text, text_probs = inference(model, input_text, infer_audio_content)
        
        audio_type = text.strip()
        audio_type = CODE_MAP.get(audio_type, '未知类别')
        fo.write(f'{key}\t{text_probs}\t{audio_type}\n')
        fo.flush()
    fo.close()

if __name__ == "__main__":
    data_input = sys.argv[1]
    infer_fo = sys.argv[2]
    model_path = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v3.3/infer_model'
    # data_input = ''
    # # csv文件；至少包含wavname、wavpath、text三列表头；分隔符：制表符('\t')
    # # 示例：/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/audio_detect/test/label_only_abnormal.csv
    # infer_fo = ''

    main_abnormal_dataset(
        model_path,
        data_input,
        infer_fo
    )
