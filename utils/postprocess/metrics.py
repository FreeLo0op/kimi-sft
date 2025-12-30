import os
import re
import json
import numpy as np
import Levenshtein
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from tn.english.normalizer import Normalizer as EN_Normalizer 
import warnings
warnings.filterwarnings("ignore")
en_tn_model = EN_Normalizer(overwrite_cache=False)

def align_accuracy(align_label_pred, align_label_true, threshold:int=120, logger=None):
    """
    计算对齐准确率。
    
    参数:
        align_label_pred (dict): 预测的对齐标签，格式为 {key: [[word, start, end], ...]}
        align_label_true (dict): 真实的对齐标签，格式为 {key: [[word, start, end], ...]}
        threshold (int): 时间阈值
    返回:
        float: 对齐准确率
    """
    all_acc_raido = []
    for key in align_label_pred:
        if key in align_label_true:
            acc_count = 0
            true_labels = align_label_true[key]
            pred_labels = align_label_pred[key]
            
            for item_pred, item_true in zip(pred_labels, true_labels):
                word_pred, start_pred, end_pred = item_pred
                word_true, start_true, end_true = item_true
                word_pred, word_true = word_pred.lower(), word_true.lower()  # 忽略大小写

                # if word_pred == word_true and abs(start_pred - start_true) <= threshold and abs(end_pred - end_true) <= threshold:
                if word_pred == word_true and start_true - threshold < start_pred and end_true + threshold > end_pred:
                    acc_count += 1
                else:
                    pass
                    # print(f"对齐错误: 预测: {item_pred}, 真实: {item_true}")
            
            acc_radio = acc_count/(len(true_labels))
            all_acc_raido.append(acc_radio)

    align_acc_mean = np.mean(all_acc_raido)
    return align_acc_mean

def align_accuracy_lcs(align_label_pred, align_label_true, threshold:int=120, logger=None):
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

def pcc_compute(true_list:list, pred_list:list, dataset_name:str=None, logger=None, if_skip:bool=False):
    """
    Compute the Pearson correlation coefficient (PCC) between two lists.
    """
    assert len(true_list) == len(pred_list), f"Length mismatch: {len(true_list)} != {len(pred_list)}"
    if len(true_list) == 0:
        return 0
    # logger.info(f"真实标签长度: {len(true_list)}, 预测标签长度: {len(pred_list)}")
    
    # Convert elements to float
    true_list = [float(x) for x in true_list]
    pred_list = [float(x) for x in pred_list]
    
    true_list = np.array(true_list)
    pred_list = np.array(pred_list)

    if np.all(true_list == true_list[0]) and np.all(pred_list == pred_list[0]):
        pcc = 1.0 if np.all(true_list == pred_list) else 0.0
    
    pcc = np.corrcoef(true_list, pred_list)[0, 1]
    if if_skip:
        return pcc
    # 使用logger记录结果
    if logger:
        logger.log_metric_result("PCC", pcc, dataset_name)
        try:
            logger.info("混淆矩阵:")
            cm = confusion_matrix(true_list, pred_list)
            for line in np.array2string(cm).splitlines():
                logger.info(line)
        except Exception as e:
            pass
    else:
        if dataset_name:
            print(f"[{dataset_name}] PCC: {pcc:.4f}")
        else:
            print(f"PCC: {pcc:.4f}")
        try:
            print(classification_report(true_list, pred_list))
            print(confusion_matrix(true_list, pred_list))
        except Exception as e:
            pass

    return pcc


def cal_hit_rate(gt_scores, pred_scores, threshold_gt=6, threshold_pred=60, logger=None):
    """
    计算命中率相关指标
    
    参数:
        gt_scores: 真实分数
        pred_scores: 预测分数
        threshold_gt: 真实分数阈值
        threshold_pred: 预测分数阈值
        logger: 日志记录器
    """
    gt_cnt = [1 if x >= threshold_gt else 0 for x in gt_scores]
    pred_cnt = [1 if x >= threshold_pred else 0 for x in pred_scores]
    hit_cnt = [1 if a and b else 0 for a, b in zip(gt_cnt, pred_cnt)]
    metrics = {
        'threshold_gt':threshold_gt,
        'threshold_pred':threshold_pred,
        'pred_count':sum(pred_cnt),
        'label_count':sum(gt_cnt),
        'hit_count':sum(hit_cnt),
        'hit_rate':sum(hit_cnt)/sum(gt_cnt),
        'recall':recall_score(gt_cnt, pred_cnt),
        'precision':precision_score(gt_cnt, pred_cnt),
        'f1':f1_score(gt_cnt, pred_cnt),
        'acc':accuracy_score(gt_cnt, pred_cnt)
    }
    
    # 使用logger记录结果
    if logger:
        logger.info("multiple_choices_metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    else:
        print(f"multiple_choices_metrics: {json.dumps(metrics, indent=2)}")
    
    return metrics


def wer(label, predict):
    # 去除标点符号
    label = text_normalize(label)
    predict = text_normalize(predict)

    # distance = Levenshtein.distance(' '.join(label), ' '.join(predict))
    distance = Levenshtein.distance(label, predict)
    wer_score = distance / len(label) if len(label) > 0 else 0
    wer_score = 1 if wer_score > 1 else wer_score  # 防止异常值
    if wer_score > 0.3:
        # print(f"高字错率警告: {wer_score:.2f}\n真实: {' '.join(label)} \n预测: {' '.join(predict)}")
        pass
    return wer_score



def mse(true_list:list, pred_list:list, dataset_name:str=None, logger=None):
    """
    Compute the Mean Squared Error (MSE) between two lists.
    """
    assert len(true_list) == len(pred_list), f"Length mismatch: {len(true_list)} != {len(pred_list)}"
    true_list = [float(x) for x in true_list]
    pred_list = [float(x) for x in pred_list]
    mse = np.mean((np.array(true_list) - np.array(pred_list)) ** 2)
    if logger:
        logger.log_metric_result("MSE", mse, dataset_name)
    else:
        print(f"MSE: {mse:.4f}")
    return mse

def acc_with_threshold(true_list:list, pred_list:list, dataset_name:str=None, threshold=1, logger=None):
    true_list = [float(x) for x in true_list]
    pred_list = [float(x) for x in pred_list]
    count = 0
    all_count = len(true_list)
    for score_pred, score_true in zip(pred_list, true_list):
        if abs(score_pred - score_true) <= threshold:
            count += 1
    acc = round(count / all_count, 2)
    if logger:
        logger.log_metric_result("Accuracy (threshold = " + str(threshold) + ")", acc, dataset_name)
    else:
        print(f"Accuracy (threshold = {threshold}) : {acc:.2f}")
    return acc

def text_normalize(text):
        if isinstance(text, list):
            text = ' '.join(text)
        # print('原始文本:', text)
        text = text.strip()
        text = text.lower()
        # text = en_tn_model.normalize(text)
        text = re.sub(r'[。，、！？：；“”‘’（）【】《》—\,\.\?\!\[\]\\\(\)\'\-]', ' ', text)
        # 中文字符前后加空格
        text = re.sub(r'([\u4e00-\u9fa5])', r' \1 ', text)
        # 多个空格合并为一个
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n', '', text)
        # print('处理后文本:', text)
        return text.split()