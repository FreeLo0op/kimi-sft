import os
import ast
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


CODE_MAP = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10}
AUDIO_TYPE_MAP = {
        '0' : '正常',
        '1' : '噪声',
        '2' : '不相关中文',
        '3' : '不相关英文',
        '4' : '无意义语音',
        '5' : '音量小',
        '6' : '开头发音不完整',
        '7' : '空音频',
        '8' : '多说话人',
    }

def llm_score_remap(infer_res:str, data_info:pd.DataFrame):
    scores = {}
    with open(infer_res, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip().split('\t')
            try:
                key, score = line[0], line[1]
                if CODE_MAP.get(score) is not None:
                    score = CODE_MAP[score]
                score = float(score)
                scores[key] = score
            except Exception as e:
                print(f"Error processing line: {line}")
    # change score2 to scores's value
    data_info['llm_score'] = data_info['wavname'].apply(lambda x: scores.get(x, -1))
    del data_info['score2']
    data_info = data_info.rename(columns={'llm_score': 'score2'})
    return data_info

def statistical_indicators_compute(data_info:pd.DataFrame, col_name:str='score2'):
    if col_name == 'score1':
        data_info['score1'] = data_info['score1'].astype(float) / 10.0
    data_info = data_info[data_info['audio_type'] != '0']
    # overall_pearson_corr = np.corrcoef(data_info['label'].tolist(), data_info[col_name].tolist())[0, 1]
    # overall_mae = np.mean(np.abs(data_info['label'] - data_info[col_name]))
    print(f"Overall Sample Size: {len(data_info)}")
    # print(f"Overall Pearson Correlation: {overall_pearson_corr:.4f}")
    # print(f"Overall MAE: {overall_mae:.4f}")

    data_type1 = data_info[data_info['audio_type'].isin(['1','5','6','8'])]
    data_type2 = data_info[data_info['audio_type'].isin(['2','3','4','7'])]
    # for subset, type_desc in zip([data_type1, data_type2], ['Type 1 (Noise, Low Volume, Incomplete Start, Multiple Speakers)', 'Type 2 (Irrelevant Chinese, Irrelevant English, Nonsense Speech, Empty Audio)']):
    for subset, type_desc in zip([data_type1], ['Type 1 (Noise, Low Volume, Incomplete Start, Multiple Speakers)']):
        if subset.empty:
            print(f"{type_desc} - No data available.")
            continue
        pearson_corr = np.corrcoef(subset['label'].tolist(), subset[col_name].tolist())[0, 1]
        mae = np.mean(np.abs(subset['label'] - subset[col_name]))
        print(f"{type_desc} Sample Size: {len(subset)}")
        print(f"  Pearson Correlation: {pearson_corr:.4f}")
        print(f"  MAE: {mae:.4f}")
    # 拒识
    data_type2['reject'] = (data_type2['label'] == 0) & (data_type2[col_name] == 0)
    # data_type2['reject'] = (data_type2[col_name] == 0)
    reject_rate = data_type2['reject'].sum() / len(data_type2)
    print(f"Type 2 Rejection Rate (label=0 and score2=0, sample size: {len(data_type2)}): {reject_rate:.4f}")

    for audio_code, type_name in AUDIO_TYPE_MAP.items():
        subset = data_info[data_info['audio_type'] == audio_code]
        # if audio_code in ['2','3','4','7']:
        #     subset = subset[subset['label'] == 0]
        if subset.empty:
            # print(f"Audio Type: {type_name} (Code: {audio_code}) - No data available.")
            continue
        print(f"Audio Type: {type_name} (Code: {audio_code}) Sample Size: {len(subset)}")
        if audio_code in ['2','3','4','7']:
            reject_rate = ((subset['label'] == 0) & (subset[col_name] == 0)).sum() / len(subset)
            print(f"  Rejection Rate (label=0 and score2=0): {reject_rate:.4f}")
        else:
            pearson_corr = np.corrcoef(subset['label'].tolist(), subset[col_name].tolist())[0, 1]
            mae = np.mean(np.abs(subset['label'] - subset[col_name]))
            print(f"  Pearson Correlation: {pearson_corr:.4f}")
            print(f"  MAE: {mae:.4f}")

# label
# data_info_path = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/audio_detect/test/label_only_abnormal_v2.csv'
data_info_path = sys.argv[1]
data_info = pd.read_csv(data_info_path, sep='\t', dtype={'wavname': str, 'text': str, 'pron_score': float, 'audio_type': str})
data_info['score2'] = -1

# 交集数据
intersect_df_path = sys.argv[2]
intersect_df = pd.read_csv(intersect_df_path, sep='\t')
data_info = data_info[data_info['wavname'].isin(intersect_df['wavname'].tolist())]

# 推理结果
new_infer_res = sys.argv[3]

data_info_new = data_info.copy()
data_info_new = llm_score_remap(new_infer_res, data_info_new)
data_info_new = data_info_new[data_info_new['score2'] != -1]
statistical_indicators_compute(data_info_new, 'score2')