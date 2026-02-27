'''
This script is used to evaluate the LLM performance.
Date: 2026-02-26
'''
import os
import re
import ast
import jiwer
import numpy as np
import pandas as pd
import logging
from typing import List
from tn.english.normalizer import Normalizer as EnNormalizer
from concurrent.futures import ThreadPoolExecutor

REF_FILES = {
    'librispeech': '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/open_source/LibriSpeech/test/label_asr',
    'aishell': '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/cn/open_source/aishell/test/test_aishell1/text',
}

PA_LABEL_FILES = {
    'next2300': '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/api_data/next/tal-k12/test/label_sent_score',
    'batch1': '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/api_data/next/tal-k12/test/label_snt_score_batch1',
    'batch2': '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/api_data/next/tal-k12/test/label_snt_score_batch2',
    'batch4': '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/api_data/next/tal-k12/test/label_snt_score_batch4',
}

PA_CODE_MAP = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

normalizer = EnNormalizer(overwrite_cache=False)

def load_text(file:str) -> pd.DataFrame:
    df = pd.read_csv(file, sep='\t', header=None, names=['key', 'text'])
    return df


def read_label_file(label_file:str, if_word:bool=False):
    label_dict = {}
    with open(label_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        if re.findall('[0-9]', lines[0]) == []:
            lines = lines[1:]
        for line in lines:
            try:
                line = line.strip().split('\t')
                try:
                    key, score = line[0], float(line[1])
                except:
                    key, score = line[0], float(line[2])
                if if_word:
                    key, score = line[0], line[2]
                    score = score.split(' ')
                    score = [float(s) for s in score]
                label_dict[key] = score
            except Exception as e:
                continue
    return label_dict


def score_scale(score:float, probability:float) -> float:
    if score == 0:
        score = 0.0 + (1 - probability) * 5
    elif score == 10:
        score =  95 + probability * 5
    else:
        score = (score * 10 - 5) + probability * 10
    return round(score, 2)


def read_pa_pred_file(pred_file:str) -> dict:
    pred_dict = {}
    with open(pred_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split('\t')
            # try:
            key, score, porbabilities = line[0], line[1], line[2]
            # key, score, porbabilities = line[0], line[1], '[1]'
            if PA_CODE_MAP.get(score) is not None:
                score = float(PA_CODE_MAP.get(score, -1))
            else:
                score = float(score)
            if score == -1:
                logger.warning(f"Warning: Key {key} has an invalid score '{score}' that cannot be mapped. Convert to 0.")
                score = 0
            porbabilities = ast.literal_eval(porbabilities)
            probality = float(porbabilities[0])
            score_scaled = score_scale(score, probality)
            pred_dict[key] = (score, score_scaled)
    return pred_dict


def pcc_compute(true_list:list, pred_list:list, dataset_name:str=None):
    """
    Compute the Pearson correlation coefficient (PCC) between two lists.
    """
    assert len(true_list) == len(pred_list), f"Length mismatch: {len(true_list)} != {len(pred_list)}"
    logger.info(f"真实标签长度: {len(true_list)}, 预测标签长度: {len(pred_list)}")
    
    # Convert elements to float
    true_list = [float(x) for x in true_list]
    pred_list = [float(x) for x in pred_list]
    
    true_list = np.array(true_list)
    pred_list = np.array(pred_list)

    if np.all(true_list == true_list[0]) and np.all(pred_list == pred_list[0]):
        pcc = 1.0 if np.all(true_list == pred_list) else 0.0
    
    pcc = np.corrcoef(true_list, pred_list)[0, 1]

    if dataset_name:
        logger.info(f"[{dataset_name}] PCC: {pcc:.4f}")
    else:
        logger.info(f"PCC: {pcc:.4f}")
    return pcc


def normalize_text(text):
    # 只保留英文、空格、中文、数字
    # 转小写
    normalized_text = normalizer.normalize(str(text))
    normalized_text = normalized_text.lower()
    normalized_text = re.sub(r'[^a-z\u4e00-\u9fa50-9 ]', '', normalized_text)
    # 中文前后加空格 用于jiwer计算WER
    normalized_text = re.sub(r'([\u4e00-\u9fa5])', r' \1 ', normalized_text)
    # 多个空格合并为一个
    normalized_text = re.sub(r' +', ' ', normalized_text)
    # 去除首尾空格
    normalized_text = normalized_text.strip()
    return normalized_text


def safe_wer(ref, hyp):
    if not ref.strip():
        return 0.0 if not hyp.strip() else 1.0
    wer = jiwer.wer(ref, hyp)
    # wer = max(0.0, min(wer, 1.0))
    return wer


def wer_evaluation(ref_text: List[str], hyp_text: List[str]):
    with ThreadPoolExecutor(max_workers=10) as executor:
        ref_normalized = list(executor.map(normalize_text, ref_text))
        hyp_normalized = list(executor.map(normalize_text, hyp_text))
    overall_wer = jiwer.wer(ref_normalized, hyp_normalized)
    logger.info(f"Overall WER: {overall_wer*100:.2f}%")

    single_wer = [
        safe_wer(ref, hyp) for ref, hyp in zip(ref_normalized, hyp_normalized)
    ]
    average_wer = np.mean(single_wer)
    logger.info(f"Average WER: {average_wer*100:.2f}%")


def main(
    ref_file:str,
    hyp_file:str,
    ):
    ref_df = load_text(ref_file)
    hyp_df = load_text(hyp_file)

    df = pd.merge(ref_df, hyp_df, on='key', how='left')
    df.rename(columns={'text_x': 'ref_text', 'text_y': 'hyp_text'}, inplace=True)
    keys_nan = df[df['hyp_text'].isna()]['key'].unique().tolist()
    if keys_nan:
        logger.warning(f'Keys with missing values: {keys_nan}')
    logger.warning(f'Total number of missing values: {len(keys_nan)}/{len(df)}')
    
    df.fillna('', inplace=True)

    wer_evaluation(df['ref_text'].tolist(), df['hyp_text'].tolist())


def snt_main(
    label_file: str,
    pred_file: str
    ):
    if not os.path.exists(pred_file):
        return
    if os.path.getsize(pred_file) == 0:
        return
    label_dict = read_label_file(label_file)
    pred_dict = read_pa_pred_file(pred_file)
    true_list, pred_list, pred_scaled_list = [], [], []
    for key in label_dict:
        if key in pred_dict:
            pred_snt_score, pred_snt_score_scaled = pred_dict[key]
            true_snt_score = label_dict[key]
            true_list.append(true_snt_score)
            pred_list.append(pred_snt_score)
            pred_scaled_list.append(pred_snt_score_scaled)
    dataset_neme = label_file.split('/')[-1]
    pcc_compute(true_list, pred_list, dataset_name=dataset_neme)
    pcc_compute(true_list, pred_scaled_list, dataset_name=f"{dataset_neme}_scaled")


if __name__ == "__main__":
    
    dataset_name = 'aishell'
    # main(
    #     ref_file = REF_FILES[dataset_name],
    #     # hyp_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/Base_Model/Kimi-PA-Base-v3/CPT_STAGE1_MODEL_0211/infer_model_ckpt23208/infer_res/libri_asr.csv'
    #     hyp_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/Base_Model/Kimi-PA-Base-v3/CPT_STAGE2_MODEL/infer_model_ckpt40000/infer_res/aishell_asr.csv'
    # )

    main(
        ref_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/audio_detect/test/noise_asr_text',
        hyp_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/Base_Model/Kimi-PA-Base-v3/CPT_STAGE2_MODEL/infer_model_ckpt40000/infer_res/noise_asr_testdataset.csv'
    )

    # snt_main(
    #     PA_LABEL_FILES['next2300']  ,
    #     pred_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/Base_Model/Kimi-PA-Base-v3/CPT_STAGE2_MODEL/infer_model_ckpt40000/infer_res/label_sent_score'
    # )
    # snt_main(
    #     PA_LABEL_FILES['batch1'],
    #     pred_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/Base_Model/Kimi-PA-Base-v3/CPT_STAGE2_MODEL/infer_model_ckpt40000/infer_res/label_snt_score_batch1'
    # )
    # snt_main(
    #     PA_LABEL_FILES['batch2'],
    #     pred_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/Base_Model/Kimi-PA-Base-v3/CPT_STAGE2_MODEL/infer_model_ckpt40000/infer_res/label_snt_score_batch2'
    # )
    # snt_main(
    #     PA_LABEL_FILES['batch4'],
    #     pred_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/Base_Model/Kimi-PA-Base-v3/CPT_STAGE2_MODEL/infer_model_ckpt40000/infer_res/label_snt_score_batch4'
    # )