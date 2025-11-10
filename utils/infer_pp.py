import os
import re
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import classification_report, confusion_matrix


def pcc_compute(true_list:list, pred_list:list, dataset_name:str=None):
    """
    Compute the Pearson correlation coefficient (PCC) between two lists.
    """
    assert len(true_list) == len(pred_list), f"Length mismatch: {len(true_list)} != {len(pred_list)}"
    # logger.info(f"真实标签长度: {len(true_list)}, 预测标签长度: {len(pred_list)}")
    
    # Convert elements to float
    true_list = [float(x) for x in true_list]
    pred_list = [float(x) for x in pred_list]
    
    true_list = np.array(true_list)
    pred_list = np.array(pred_list)

    if np.all(true_list == true_list[0]) and np.all(pred_list == pred_list[0]):
        pcc = 1.0 if np.all(true_list == pred_list) else 0.0
    
    pcc = np.corrcoef(true_list, pred_list)[0, 1]

    if dataset_name:
        print(f"[{dataset_name}] PCC: {pcc:.2f}")
    else:
        print(f"PCC: {pcc:.2f}")
    try:
        print(classification_report(true_list, pred_list))
        print(confusion_matrix(true_list, pred_list))
    except Exception as e:
        pass

    return pcc

def align_accuracy_lcs(align_label_pred, align_label_true, threshold:int=120):
    """
    计算对齐准确率（v2）。
    - 允许预测与标注长度不一致，使用 LCS（自定义匹配条件）进行序列对齐
    - 匹配条件与 v1 保持一致：词相同且预测区间落在阈值扩展的真实区间内
    - 每个样本的准确率 = 最优匹配数 / len(true_labels)，最后取均值
    """
    all_acc_ratio = []

    for key, pred_labels in align_label_pred.items():
        if key not in align_label_true:
            continue

        true_labels = align_label_true[key]
        n, m = len(true_labels), len(pred_labels)
        if n == 0:
            continue

        # dp[i+1][j+1] 表示 true 前 i 个与 pred 前 j 个的 LCS 长度
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        def is_match(i, j):
            word_true, start_true, end_true = true_labels[i]
            word_pred, start_pred, end_pred = pred_labels[j]
            wt = str(word_true).lower()
            wp = str(word_pred).lower()
            if wt != wp:
                return False
            return (start_true - threshold) < start_pred and (end_true + threshold) > end_pred

        for i in range(n):
            for j in range(m):
                if is_match(i, j):
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    left = dp[i + 1][j]
                    up = dp[i][j + 1]
                    dp[i + 1][j + 1] = left if left >= up else up

        lcs_len = dp[n][m]
        acc = lcs_len / n
        all_acc_ratio.append(acc)

    return np.mean(all_acc_ratio)

def snt_infer_pp(pred_res:str):
    """
    处理预测结果，提取预测分数
    """
    label_score_list, pred_score_list = [], []
    pattern = r'结果为：([0-9\.]+)。'

    with open(pred_res, 'r', encoding='utf-8') as fin:
        for line in fin:
            infer_res = json.loads(line.strip())
            label = infer_res.get('label', None)
            pred = infer_res.get('predict', None)

            if pred is not None:
                match = re.search(pattern, pred)
                if match:
                    pred_score = float(match.group(1))
                else:
                    pred_score = None
            
            if label is not None:
                match = re.search(pattern, label)
                if match:
                    label_score = float(match.group(1))
                else:
                    label_score = None
            
            if label_score is not None and pred_score is not None:
                label_score_list.append(label_score)
                pred_score_list.append(pred_score)
    
    pcc_compute(label_score_list, pred_score_list)

def phn_infer_pp(pred_res:str):
    """
    处理预测结果，提取预测分数
    """
    label_score_list, pred_score_list = [], []

    with open(pred_res, 'r', encoding='utf-8') as fin:
        for line in fin:
            infer_res = json.loads(line.strip())
            label = infer_res.get('label', None)
            pred = infer_res.get('predict', None)

            if label is not None and pred is not None:
                label = json.loads(label)
                pred = json.loads(pred)

                pred_phn_score_list = [item['score'] for item in label]
                label_phn_score_list = [item['score'] for item in pred]
                if len(pred_phn_score_list) != len(label_phn_score_list):
                    continue
                for i in range(len(pred_phn_score_list)):
                    pred_scores = pred_phn_score_list[i].split()
                    label_scores = label_phn_score_list[i].split()
                    if len(pred_scores) != len(label_scores):
                        continue
                    pred_scores = [float(x) for x in pred_scores]
                    label_scores = [float(x) for x in label_scores]
                    pred_score_list.extend(pred_scores)
                    label_score_list.extend(label_scores)

    pcc_compute(label_score_list, pred_score_list)

def word_infer_pp(pred_res:str, pa_type:str):
    """
    处理预测结果，提取预测分数
    """
    label_score_list, pred_score_list = [], []
    pattern = r'结果为：(\[.*\]+)。'

    with open(pred_res, 'r', encoding='utf-8') as fin:
        for line in fin:
            infer_res = json.loads(line.strip())
            label = infer_res.get('label', None)
            pred = infer_res.get('predict', None)

            if label is not None and pred is not None:
                try:
                    label = re.search(pattern, label).group(1)
                    pred = re.search(pattern, pred).group(1)
                    label = json.loads(label)
                    pred = json.loads(pred)
                except:
                    continue

                pred_word_score_list = [float(item[pa_type]) for item in label]
                label_word_score_list = [float(item[pa_type]) for item in pred]
                if len(pred_word_score_list) != len(label_word_score_list):
                    continue
                pred_score_list.extend(pred_word_score_list)
                label_score_list.extend(label_word_score_list)

    pcc_compute(label_score_list, pred_score_list)

def align_infer_pp(pred_res:str):
    """
    处理预测结果，提取预测分数
    """
    label_score_list, pred_score_list = [], []

    with open(pred_res, 'r', encoding='utf-8') as fin:
        for line in fin:
            infer_res = json.loads(line.strip())
            label = infer_res.get('label', None)
            pred = infer_res.get('predict', None)

            if label is not None and pred is not None:
                label = label.split()
                pred = pred.split()
                if len(label) != len(pred):
                    continue
                label = [float(x) for x in label]
                pred = [float(x) for x in pred]
                label_score_list.extend(label)
                pred_score_list.extend(pred)



if __name__ == "__main__":
    # pred_res = '/mnt/pfs_l2/jieti_team/SFT/hupeng/github/Kimi-Audio/output/infer_res/infer_speechocean762_word_pa_total_nocot-v2_test.json'
    infer_dir = '/mnt/pfs_l2/jieti_team/SFT/hupeng/github/Kimi-Audio/output/infer_res'
    for file in os.listdir(infer_dir):
        if file.endswith('.json'):
            pred_res = os.path.join(infer_dir, file)
            print(f"Processing {file} ...")

            if 'sent_pa' in pred_res:
                snt_infer_pp(pred_res)
            elif 'phoneme_pa' in pred_res:
                phn_infer_pp(pred_res)
            elif 'word_pa' in pred_res:
                if 'accuracy' in pred_res:
                    pa_type = 'accuracy'
                elif 'stress' in pred_res:
                    pa_type = 'stress'
                elif 'total' in pred_res:
                    pa_type = 'total'
                word_infer_pp(pred_res, pa_type)
