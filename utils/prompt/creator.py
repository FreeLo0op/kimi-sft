'''
base train prompt for Kimi-Audio
'''
import os
import re
import json
import torchaudio
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MAX_WORKERS = 48
MAX_AUDIO_DURATION = 29  # seconds

# global config
WAVPATH_MAP = {
    'abc-reading': '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/abc_reading/wavpath',
    'wangxiao-online': '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/wangxiao_online/wavpath',
    'next-online': '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/next_online/wavpath',
    'librispeech': '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/LibriSpeech/train/wavpath',
    'aishell2': '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/cn/aishell2/train/wavpath',
    'aishell3': '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/cn/aishell3/aishell3_clean/wavpath',
    'child-data': '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/cn/child-data/child-data-clean/labels_all/wavpath',
    'thchs30': '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/cn/thchs-30/thchs-30_clean/wavpath',
    'weilaimofa': '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/weilaimofa/20250527/wavpath',
    'ertongyanjiang': '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/cn/ertongyanjiang/ertongyanjiang_clean/wavpath'
}

TRAIN_DATASET_DIR = '/mnt/pfs_l2/jieti_team/SFT/hupeng/mdd_lm/data/multi_task/base_model_v3/train'
EVAL_DATASET_DIR = '/mnt/pfs_l2/jieti_team/SFT/hupeng/mdd_lm/data/multi_task/base_model_v3/eval'

def get_wavpath(dataset_name):
    audio_map = {}
    for key, value in WAVPATH_MAP.items():
        if key in dataset_name.lower():
            with open(value, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    audio_key, audio_path = line.strip().split('\t', maxsplit=1)
                    audio_map[audio_key] = audio_path
            return audio_map
    return audio_map

def check_audio_valid(wavpath:str):
    try:
        waveform, sample_rate = torchaudio.load(wavpath)
        duration = waveform.shape[1] / sample_rate
        if 0 < duration < MAX_AUDIO_DURATION and sample_rate == 16000:
            return True
        else:
            logger.warning(f"Invalid audio: {wavpath}, duration: {duration}, sample_rate: {sample_rate}")
            return False
    except Exception as e:
        logger.warning(f"Error loading audio: {wavpath}, error: {e}")
        return False

def prompt_convert(data, audio_map:dict={}):
    messages = data["messages"]
    system_content = messages[0]["content"]
    user_content = messages[1]["content"]
    user_content = re.sub(r'<audio>', '', user_content)
    assistant_content = messages[2]["content"]
    audio = data["audios"][0]
    audio_key = os.path.basename(audio)
    audio_key = re.sub(r'\.flac\.wav$|\.wav$', '', audio_key)

    if audio_map:
        new_audio = audio_map.get(audio_key, None)
        if new_audio:
            audio = new_audio

    if check_audio_valid(audio):
        data = {
            "task_type": "understanding",
            "conversation":[
                {
                    "role": "user",
                    "message_type": "text",
                    "content": f"{system_content}{user_content}"
                },
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": audio
                },
                {
                    "role": "assistant",
                    "message_type": "text",
                    "content": assistant_content
                }
            ]
        }
    else:
        data = None

    return data

def _safe_convert(old_prompt):
    try:
        return prompt_convert(old_prompt)
    except Exception as exc:  # 捕获转换过程中的异常，避免阻断整体流程
        logger.warning(f"Prompt conversion failed {exc}")
        return None

def main_base(
        dataset_dir:str,
        save_dir:str
):
    for dataset in os.listdir(dataset_dir):
        if dataset.endswith('json'):
            dataset_path = os.path.join(dataset_dir, dataset)
            output_path = os.path.join(save_dir, dataset)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"File exists, skip: {output_path}")
                continue
            new_prompts = []
            counter = 0
            logger.info(f"Processing dataset: {dataset_path}")
            with open(dataset_path, 'r', encoding='utf-8') as f:
                old_prompts = json.load(f)
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    for new_prompt in tqdm(
                        executor.map(_safe_convert, old_prompts),
                        total=len(old_prompts),
                        desc=f'Processing : {dataset}'
                    ):
                        if new_prompt:
                            new_prompts.append(new_prompt)
                            counter += 1

            logger.info(f"Processed {counter}/{len(old_prompts)} valid prompts from {dataset}, {len(old_prompts)-counter} invalid prompts are removed.")

            save_root = os.path.dirname(output_path)
            if not os.path.exists(save_root):
                os.makedirs(save_root, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in new_prompts:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    
        # 测试中断
        # break

def main_sft(
        dataset_type:str,
        save_path:str
        ):
    train_root_dir = '/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/multi_task/sft/train'
    train_ori_prompts = [
        # 对齐
        'next-online-10k_align_cotv2_train.json', 'ertonggushi_align_cotv1_train.json',
        # 音素评测
        'speechocean762_phoneme_withcmu_pa_cotv2_train.json', 
        'tal-k12_phoneme_ipa_pa_cotv2_train.json',
        'tal-k12_phoneme_withipa_pa_cotv2_train.json',
        'tal-k12_phoneme_recommend_cotv2_train.json',
        # 单词评测
        'speechocean762_word_pa_accuracy_nocot-v2_train.json', 'speechocean762_word_pa_total_nocot-v2_train.json', 'tal-k12_word_pa_accuracy_nocot-v2_train.json',
        # 句子评测
        'speechocean762_sent_pa_accuracy_nocot-v2_train.json', 'speechocean762_sent_pa_fluency_nocotv1_train.json', 'speechocean762_sent_pa_prosodic_nocot-v2_train.json', 'speechocean762_sent_pa_total_nocot-v2_train.json', 'tal-k12_sent_pa_accuracy_nocot-v2_train.json', 'next_sent_pa_fluency_nocotv1_train.json', 'xiaohou_sent_pa_accuracy_nocotv3_train.json',
        # 音素、单词、句子三维评测
        'tal-k12_full_pa_llmgt_cotv1_train.json',
        # 连读检测
        'ldtrain_liaison_detection_nocotv1_train.json',
        # 多句多选
        'bdy_wx_0506_multiple_choice_cotv1_train.json',
        # 重复读检测
        'wangxiao-online-20250527_repeat_reading_cotv1_train.json',
        # 识别
        'librispeech_asr_cotv1_train.json', 'ertonggushi_asr_cotv1_train.json',
        # KET
        # 'ket_pa_v1_train.json',
        # 学习机半开放题
        'xxj_compare_v2_train.json'
    ]

    eval_root_dir = '/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/multi_task/sft/eval'

    if dataset_type == 'train':
        root_dir = train_root_dir
        # ori_prompts = train_ori_prompts
    elif dataset_type == 'eval':
        root_dir = eval_root_dir
        # ori_prompts = [item.replace('train.json', 'eval.json') for item in train_ori_prompts]
    ori_prompts = os.listdir(root_dir)
    new_prompts = []
    for ori_prompt in ori_prompts:
        if 'speechocean762' in ori_prompt:
            continue
        prompt_path = os.path.join(root_dir, ori_prompt)
        if not os.path.exists(prompt_path):
            logger.warning(f"File not found: {prompt_path}")
            continue
        with open(prompt_path, 'r', encoding='utf-8') as f:
            count = 0
            old_prompts = json.load(f)

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for new_prompt in tqdm(
                    executor.map(_safe_convert, old_prompts),
                    total=len(old_prompts),
                    desc=f'Processing : {ori_prompt}'
                ):
                    if new_prompt:
                        new_prompts.append(new_prompt)
                        count += 1

        logger.info(f"Processed {count}/{len(old_prompts)} valid prompts from {ori_prompt}, {len(old_prompts)-count} invalid prompts are removed.")

    # if dataset_type == 'train':
    #     ad_prompt = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/audio_detect/train/ad_train_260112_llm.jsonl'
    #     with open(ad_prompt, 'r') as fin:
    #         for line in fin:
    #             line = json.loads(line.strip())
    #             new_prompts.append(line)
        
    save_root = os.path.dirname(save_path)
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in new_prompts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main_sft(
        dataset_type='train',
        save_path='/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/sft/train/train_29.json'
    )
    main_sft(
        dataset_type='eval',
        save_path='/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/sft/dev/eval_29.json'
    )

    # main_base(
    #     dataset_dir='/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/multi_task/base_model_v4/eval',
    #     save_dir='/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/base/dev/single_datasets'
    # )