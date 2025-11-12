import os
import re
import sys
import ast
import json
import csv
import jiwer
from collections import defaultdict
from postprocess.metrics import *
import yaml
import pandas as pd

DATASET_INFO = '/mnt/pfs_l2/jieti_team/SFT/hupeng/mdd_lm/data/dataset_info.json'
DATASET_ROOOT = '/mnt/pfs_l2/jieti_team/SFT/hupeng/mdd_lm/data'

def read_prompt_json(file_path:str):
    """
    读取JSON文件并返回数据。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    
    return data

def load_infer_data(infer_config_file:str):
    with open(infer_config_file, 'r') as f:
        infer_config = yaml.safe_load(f)
    
    dataset_info_dict = defaultdict(list)
    with open(DATASET_INFO, 'r') as f:
        dataset_info = json.load(f)

    infer_dataset = infer_config['eval_dataset']
    infer_dataset = infer_dataset.split(',')
    for dataset in infer_dataset:
        dataset = dataset.strip()
        file_name = dataset_info[dataset]['file_name']
        dataset_root = os.path.join(DATASET_ROOOT, file_name)
        prompt = read_prompt_json(dataset_root)
        prompt_length = len(prompt)
        dataset_info_dict[dataset].append(prompt_length)
    
    return dataset_info_dict

def infer_res_split(infer_config_file:str, infer_res_file:str):

    dataset_info_dict = load_infer_data(infer_config_file)
    with open(infer_res_file, 'r') as f:
        lines = f.readlines()
    
    last_index = 0
    for dataset, infos in dataset_info_dict.items():
        length = infos[0]
        infer_res = lines[last_index : last_index + length]
        dataset_info_dict[dataset].append(infer_res)
        last_index += length
    
    return dataset_info_dict    

def score_check(dataset_name:str, score_level:str, score_type:str, score, logger, data_key)->float:
    score = float(score)
    if score_type:
        key = f'{dataset_name}_{score_level}_{score_type}'
    else:
        key = f'{dataset_name}_{score_level}'
    score_map = {
        'speechocean762_phoneme_accuracy': [0, 2],
        'speechocean762_word_accuracy': [0, 10],
        'speechocean762_word_total': [0, 10],
        'speechocean762_sentence': [0, 10],
        'tal-k12_phoneme_accuracy': [0, 1],
        'tal-k12_word_accuracy': [0, 3],
        'tal-k12_sentence_accuracy': [0, 10],
        'tal-k12_sentence': [0, 10],
        'tal-k12_sentence_fluency': [0, 3],
        'xiaohou_word_accuracy': [0, 3],
        'xiaoheiban_word_accuracy': [0, 3],
        'next_sentence': [0, 3],
        'sent_sentence': [0, 5],
        'open_word_accuracy': [0, 10],
    }
    if key in score_map:
        min_score, max_score = score_map[key]
        if score > max_score:
            logger.warning(f'{data_key} predict value out of range [{min_score}, {max_score}]: {score}')
            return float(max_score)
        elif score < min_score:
            logger.warning(f'{data_key} predict value out of range [{min_score}, {max_score}]: {score}')
            return float(min_score)
        else:
            return float(score)

def get_align_res(infer_infos, logger):
    total_num, infer_ress = infer_infos
    diff_num, overlap_num, pred_failed_num = 0, 0, 0
    
    align_label_pred = defaultdict(list)
    align_label_pred_same = defaultdict(list)
    align_label_true = defaultdict(list)

    for line in infer_ress:
        try:
            line = json.loads(line)
            predict, label = line['predict'], line['label']
            key, ext = os.path.splitext(os.path.basename(line['audio']))
            predict = re.findall(r'对齐结果：(\[[^\[\]]*\])', predict)
            label = re.findall(r'对齐结果：(\[[^\[\]]*\])', label)
            label = json.loads(label[0])
        except Exception as e:
            # logger.error(f"Error parsing line: {key} : {e}")
            pred_failed_num += 1
            break
        if predict:
            try:
                predict = json.loads(predict[0])
                for item in predict:
                    word, start, end = str(item['word']), int(item['start']), int(item['end'])
                    align_label_pred[key].append([word, start, end])
                    if end < start:
                        overlap_num += 1
            except json.JSONDecodeError:
                pred_failed_num += 1
                continue
            except Exception as e:
                # logger.error(f"Error parsing line: {key} : {e}")
                pred_failed_num += 1
                continue

            for item in label:
                align_label_true[key].append([item['word'], item['start'], item['end']])

            pred_snt = ' '.join([item[0] for item in align_label_pred[key]])
            true_snt = ' '.join([item[0] for item in align_label_true[key]])
            if pred_snt.lower() == true_snt.lower():
                align_label_pred_same[key] = align_label_pred[key]
            else:
                logger.warning(f"Alignment mismatch for {key}: {pred_snt} != {true_snt}")
                diff_num += 1
        else:
            pred_failed_num += 1
    
    align_acc_mean = align_accuracy(align_label_pred, align_label_true, threshold=120, logger=logger)
    align_acc_mean_same = align_accuracy(align_label_pred_same, align_label_true, threshold=120, logger=logger)
    
    # 使用logger记录结果
    logger.log_prediction_stats(pred_failed_num, total_num, diff_num)
    logger.info(f"对齐时间戳 overlap 数量: {overlap_num} / {total_num} ({overlap_num/total_num:.2%})")   
    logger.info(f"对齐结果 missmatch 数量: {diff_num} / {total_num} ({diff_num/total_num:.2%})")
    logger.info(f"对齐准确率: {align_acc_mean:.4f}")
    logger.info(f"预测文本和标签完全一致数量: {len(align_label_pred_same)} / {total_num} ({len(align_label_pred_same)/total_num:.2%})")
    logger.info(f"预测文本和标签完全一致对齐准确率: {align_acc_mean_same:.4f}")

    # 使用 v2 对齐准确率
    align_acc_mean_v2 = align_accuracy_lcs(align_label_pred, align_label_true, threshold=120, logger=logger)
    align_acc_mean_same_v2 = align_accuracy_lcs(align_label_pred_same, align_label_true, threshold=120, logger=logger)
    logger.info(f"对齐准确率 v2: {align_acc_mean_v2:.4f}")
    logger.info(f"预测文本和标签完全一致对齐准确率 v2: {align_acc_mean_same_v2:.4f}")

def get_phoneme_pa_res(infer_infos, dataset_name ,logger):
    total_num, infer_ress = infer_infos
    pred_failed_num, diff_num = 0, 0
    phoneme_label_pred, phoneme_label_true = list(), list()

    for line in infer_ress:
        line = json.loads(line)
        predict, label = line['predict'], line['label']
        key, ext = os.path.splitext(os.path.basename(line['audio']))
    
        try:
            predict = json.loads(predict)
            label = json.loads(label)
            pred_scores = [item['score'] for item in predict]
            label_scores = [item['score'] for item in label]
            if len(pred_scores) != len(label_scores):
                diff_num += 1
                continue
        except Exception as e:
            # logger.error(f"Error parsing line: {key} : {e}")
            pred_failed_num += 1
            continue

        try:
            utt_pred_scores, utt_label_scores = [], []
            for pred_item, label_item in zip(pred_scores, label_scores):
                
                pred_item = pred_item.split()
                pred_item = [float(item) for item in pred_item]
                # pred_item = [item if item <= 2.0 else 2.0 for item in pred_item]
                for i in range(len(pred_item)):
                    pred_item[i] = score_check(dataset_name, 'phoneme', 'accuracy', pred_item[i], logger, key)

                label_item = label_item.split()
                utt_pred_scores.extend(pred_item)
                utt_label_scores.extend(label_item)

            if len(utt_label_scores) != len(utt_pred_scores):
                diff_num += 1
            else:
                phoneme_label_pred.extend(utt_pred_scores)
                phoneme_label_true.extend(utt_label_scores)
        except Exception as e:
            # logger.error(f"Error parsing line: {key} : {e}")
            pred_failed_num += 1
            continue
    
    # 使用logger记录结果
    logger.log_prediction_stats(pred_failed_num, total_num, diff_num)
    pcc_compute(phoneme_label_true, phoneme_label_pred, logger=logger)

def get_word_pa_res(infer_infos, pa_type:str, dataset_name:str, logger):
    total_num, infer_ress = infer_infos
    pred_failed_num, diff_num = 0, 0
    word_label_pred, word_label_true = list(), list()
    word_label_modify = {}
    with open('/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_word_accuracy_modify', 'r') as f:
        for line in f:
            line = line.strip().split('\t', maxsplit=2)
            key = line[0]
            word_label_modify[key] = line[2].split(' ')
    logger.info(f"评估类型: {pa_type}")

    for line in infer_ress:
        line = json.loads(line)
        predict, label, audio = line['predict'], line['label'], line['audio']
        key, ext = os.path.splitext(os.path.basename(audio))

        try:
            if dataset_name == 'xiaoheiban':
                predict = re.findall(r'结果为：(.*)。', predict)[0]
                label = re.findall(r'结果为：(.*)。', label)[0]
                pred_scores = ast.literal_eval(predict)
                label_scores = ast.literal_eval(label)
            else:
                predict = re.findall(r'结果为：(\[[^\[\]]*\])', predict)[0]
                label = re.findall(r'结果为：(\[[^\[\]]*\])', label)[0]
                predict = json.loads(predict)
                label = json.loads(label)
                pred_scores = [item[pa_type] for item in predict]
                label_scores = [item[pa_type] for item in label]
                if key in word_label_modify:
                    label_scores = word_label_modify[key]
                words = [item['word'] for item in predict]
        except Exception as e:
            # logger.error(f"Error parsing predict: {key} : {e}")
            pred_failed_num += 1
            continue

        if len(pred_scores) != len(label_scores):
            diff_num += 1
            continue
        try:
            pred_scores = [float(score) for score in pred_scores]
            label_scores = [float(score) for score in label_scores]
            
            # for i in range(len(pred_scores)):
            #     pred_scores[i] = score_check(dataset_name, 'word', 'accuracy', pred_scores[i], logger, key)
        except Exception as e:
            logger.error(f"Error parsing predict: {key} : {e}")
            pred_failed_num += 1
            continue
        

        word_label_pred.extend(pred_scores)
        word_label_true.extend(label_scores)

    # 使用logger记录结果
    logger.log_prediction_stats(pred_failed_num, total_num, diff_num)
    pcc_compute(word_label_true, word_label_pred, dataset_name, logger=logger)

    acc_threshold_1 = acc_with_threshold(word_label_true, word_label_pred, threshold=1)
    acc_threshold_0 = acc_with_threshold(word_label_true, word_label_pred, threshold=0)
    logger.info(f'准确率(1分容错): {acc_threshold_1:.4f}')
    logger.info(f'准确率(0分容错): {acc_threshold_0:.4f}')


def get_snt_pa_res(infer_infos, dataset_name, logger):
    total_num, infer_ress = infer_infos
    pred_failed_num = 0
    snt_label_pred, snt_label_true = list(), list()
    for line in infer_ress:
        try:
            line = json.loads(line)
            predict, label = line['predict'], line['label']
            key, ext = os.path.splitext(os.path.basename(line['audio']))
            predict = re.findall(r'结果为：(.*)。', predict)[0]
            predict = float(predict)
            predict = score_check(dataset_name, 'sentence', None, predict, logger, key)
            label = re.findall(r'结果为：(.*)。', label)[0]
            label = float(label)

            snt_label_pred.append(predict)
            snt_label_true.append(label)

        except Exception as e:
            # logger.error(f"Error parsing predict: {key} : {e}")
            pred_failed_num += 1
            continue

    # 使用logger记录结果
    logger.log_prediction_stats(pred_failed_num, total_num)
    pcc_compute(snt_label_true, snt_label_pred, dataset_name, logger=logger)

def get_snt_multi_pa_res(infer_infos, dataset_name, logger):

    def extract_scores(text):
        accuracy_score = None
        fluency_score = None
        try:
            json_match = re.findall(r'结果为：(.*)。', text)
            if json_match:
                json_str = json_match[0]
                try:
                    score_data = json.loads(json_str)
                    accuracy_score = float(score_data.get('accuracy', 0))
                    fluency_score = float(score_data.get('fluency', 0))
                    fluency_score = min(fluency_score, 3)
                    fluency_score = max(fluency_score, 0)
                    accuracy_score = min(accuracy_score, 10)
                    accuracy_score = max(accuracy_score, 0)
                    return accuracy_score, fluency_score
                except Exception as e:
                    pass
        except Exception as e:
            pass
        return None, None

    total_num, infer_ress = infer_infos
    pred_failed_num = 0
    snt_acc_pred, snt_flu_pred = list(), list()
    snt_acc_true, snt_flu_true = list(), list()
    for line in infer_ress:
        try:
            line = json.loads(line)
            predict, label = line['predict'], line['label']
            key, ext = os.path.splitext(os.path.basename(line['audio']))
            acc_pred, flu_pred = extract_scores(predict)
            acc_true, flu_true = extract_scores(label)
            if acc_pred is not None and flu_pred is not None:
                snt_acc_pred.append(acc_pred)
                snt_flu_pred.append(flu_pred)
                snt_acc_true.append(acc_true)
                snt_flu_true.append(flu_true)
            else:
                pred_failed_num += 1
                continue
        except Exception as e:
            pred_failed_num += 1
            continue
    logger.info(f'pred_failed_num: {pred_failed_num}')
    pcc_compute(snt_acc_true, snt_acc_pred, f'{dataset_name}:Accuracy', logger=logger)
    pcc_compute(snt_flu_true, snt_flu_pred, f'{dataset_name}:Fluency', logger=logger)

def get_liaison_res(infer_infos, logger):
    total_num, infer_ress = infer_infos
    true_labels, pred_labels = [], []
    tf_true_labels, tf_pred_labels = [], []
    diff_len_num, failed_num = 0, 0

    for line in infer_ress:
        try:
            line = json.loads(line)
            label_ref = line['label']
            label_ref = re.findall(r'发音检测结果为：(.*)。', label_ref)
            label_ref = ast.literal_eval(label_ref[0])
            
            predict_res = line['predict']
            predict_res = re.findall(r'发音检测结果为：(.*)。', predict_res)
            predict_res = ast.literal_eval(predict_res[0])
            
            if len(predict_res) != len(label_ref):
                diff_len_num += 1
                continue
            
            predict_res = [int(x) for x in predict_res]
            label_ref = [int(x) for x in label_ref]
            predict_tf_res = [1 if x > 0 else 0 for x in predict_res]
            label_tf_ref = [1 if x > 0 else 0 for x in label_ref]
            # text = re.findall(r'参考文本:(.*)<\|im_end\|>', line['prompt'])[0].split(',')

            pred_labels.extend(predict_res)
            true_labels.extend(label_ref)

            tf_pred_labels.extend(predict_tf_res)
            tf_true_labels.extend(label_tf_ref)
        except:
            failed_num += 1
            continue
            
    metrics = {
        '单标签': {
            'accuracy': accuracy_score(tf_true_labels, tf_pred_labels),
            'precision': precision_score(tf_true_labels, tf_pred_labels),
            'recall': recall_score(tf_true_labels, tf_pred_labels),
            'f1': f1_score(tf_true_labels, tf_pred_labels)
        },
        '多标签': {
            'accuracy': round(accuracy_score(true_labels, pred_labels), 2),
            'precision': ','.join([str(round(p, 2)) for p in precision_score(true_labels, pred_labels, average=None)]),
            'recall': ','.join([str(round(r, 2)) for r in recall_score(true_labels, pred_labels, average=None)]),
            'f1': ','.join([str(round(f, 2)) for f in f1_score(true_labels, pred_labels, average=None)])
        }
    }
    
    logger.info(f'liaison_metrics: {json.dumps(metrics, indent=2, ensure_ascii=False)}')
    logger.log_metric_result('Infer failed nums', failed_num / total_num, "liaison")
    logger.log_metric_result('diff_len_rate', diff_len_num / total_num, "liaison")

def get_multiple_choice_res(infer_infos, logger):
    tg_threshold, pred_threshold = 6, 6
    record = {}
    total_num, infer_ress = infer_infos
    pred_failed_num, diff_len_num = 0, 0

    total_snt_num, pre_snt_num = 0, 0
    hit_snt_num = 0
    
    total_snt_num_threshold, pre_snt_num_threshold = 0, 0
    hit_snt_num_threshold = 0

    for i in range(total_num):
        line = json.loads(infer_ress[i])
        # key = line['audio'][0].split('/')[-1].split('.')[0]
        snt_hit = 0
        try:
            label_true = line['label']
            label_true = re.findall(r'多句多选检测结果：(.*)。', label_true)
            label_true = ast.literal_eval(label_true[0])
            len_label_true = len(label_true)
            label_true = [(x['idx'], float(x['total_score'])) for x in label_true]
            total_snt_num += len(label_true)
            for _, score in label_true:
                if score >= tg_threshold:
                    total_snt_num_threshold += 1
        except Exception as e:
            # print( f"Error parsing label line: {i+1} {key}, {e}")
            continue
        
        try:
            label_pred = line['predict']
            label_pred = re.findall(r'多句多选检测结果：(.*)。', label_pred)
            label_pred = ast.literal_eval(label_pred[0])
            label_pred = [(x['idx'], float(x['total_score'])) for x in label_pred if x['start'] != x['end'] != '0']
            pre_snt_num += len(label_pred)
            for _, score in label_pred:
                if score >= pred_threshold:
                    pre_snt_num_threshold += 1
        except Exception as e:
            pred_failed_num += 1
            # logger.error(f"Error parsing predict line: {i+1}, {e}")
            continue

        if len(label_pred) != len(label_true):
            diff_len_num += 1

        for idx_pred, score_pred in label_pred:
            for idx_true, score_true in label_true[:]:
                if idx_pred == idx_true:
                    hit_snt_num += 1
                    snt_hit += 1
                    if score_pred >= pred_threshold and score_true >= tg_threshold:
                        hit_snt_num_threshold += 1
                    label_true.remove((idx_true, score_true))
                    break
        snt_hit_rate = snt_hit / len_label_true
        
        # record[key] = [snt_hit_rate, line['predict'], line['label'], line['audio'][0]]

    recall = hit_snt_num / total_snt_num if total_snt_num > 0 else 0
    recall_threshold = hit_snt_num_threshold / total_snt_num_threshold if total_snt_num_threshold > 0 else 0

    precision = hit_snt_num / pre_snt_num if pre_snt_num > 0 else 0
    precision_threshold = hit_snt_num_threshold / pre_snt_num_threshold if pre_snt_num_threshold > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_score_threshold = 2 * (precision_threshold * recall_threshold) / (precision_threshold + recall_threshold) if (precision_threshold + recall_threshold) > 0 else 0

    # 使用logger记录结果
    logger.log_prediction_stats(pred_failed_num, total_num, diff_len_num)
    logger.log_metric_result("总句子数量", total_snt_num, "multiple_choice")
    logger.log_metric_result("预测句子数量", pre_snt_num, "multiple_choice")
    logger.log_metric_result("命中句子数量", hit_snt_num, "multiple_choice")
    logger.log_metric_result("召回率", recall, "multiple_choice")
    logger.log_metric_result("精确率", precision, "multiple_choice")
    logger.log_metric_result("F1分数", f1_score, "multiple_choice")
    
    logger.log_metric_result("阈值" + str(tg_threshold) + "," + str(pred_threshold) + " 总句子数量", total_snt_num_threshold, "multiple_choice")
    logger.log_metric_result("阈值" + str(tg_threshold) + "," + str(pred_threshold) + " 预测句子数量", pre_snt_num_threshold, "multiple_choice")
    logger.log_metric_result("阈值" + str(tg_threshold) + "," + str(pred_threshold) + " 命中句子数量", hit_snt_num_threshold, "multiple_choice")
    logger.log_metric_result("阈值" + str(tg_threshold) + "," + str(pred_threshold) + " 召回率", recall_threshold, "multiple_choice")
    logger.log_metric_result("阈值" + str(tg_threshold) + "," + str(pred_threshold) + " 精确率", precision_threshold, "multiple_choice")
    logger.log_metric_result("阈值" + str(tg_threshold) + "," + str(pred_threshold) + " F1分数", f1_score_threshold, "multiple_choice")

def get_repeat_reading_res(infer_infos, logger):
    total_num, infer_ress = infer_infos
    pred_failed_num = 0
    repeat_label_pred, repeat_label_true = list(), list()
    repeat_label_pred_remove, repeat_label_true_remove = list(), list()

    for line in infer_ress:
        flag = False
        line = json.loads(line)
        predict, label = line['predict'], line['label']
        key, ext = os.path.splitext(os.path.basename(line['audio']))
        try:
            predict = re.findall(r'重复读次数为：(.*)。', predict)[0]
            label = re.findall(r'重复读次数为：(.*)。', label)[0]
            predict = float(predict)
            label = float(label)
            if label == -1.0:
                flag = True
        except json.JSONDecodeError:
            pred_failed_num += 1
            continue
        except Exception as e:
            # logger.error(f"Error parsing predict line: {key} : {e}")
            pred_failed_num += 1
            continue
        
        if not flag:
            repeat_label_pred_remove.append(predict)
            repeat_label_true_remove.append(label)
        repeat_label_pred.append(predict)
        repeat_label_true.append(label)
    
    # 使用logger记录结果
    logger.log_prediction_stats(pred_failed_num, total_num)
    pcc_compute(repeat_label_true, repeat_label_pred, logger=logger)
    logger.info(f"去除Lable为-1.0的测试样本")
    pcc_compute(repeat_label_true_remove, repeat_label_pred_remove, logger=logger)

def get_asr_res(infer_infos, logger):
    total_num, infer_ress = infer_infos
    pred_failed_num = 0
    true_labels, pred_labels = [], []
    for line in infer_ress:
        break_flage = False
        try:
            line = json.loads(line)
            key, ext = os.path.splitext(os.path.basename(line['audio']))
            predict_res = line['predict']
            predict_res = re.findall(r'识别结果为：(\[[^\[\]]*\])', predict_res)
            predict_res = ast.literal_eval(predict_res[0])
            for item in predict_res:
                if not isinstance(item, str):
                    break_flage = True
                    break
            if break_flage:
                pred_failed_num += 1
                continue
            label_ref = line['label']
            label_ref = re.findall(r'识别结果为：(\[[^\[\]]*\])', label_ref)
            label_ref = ast.literal_eval(label_ref[0])

            pred_labels.append(predict_res)
            true_labels.append(label_ref)
        except Exception as e:
            # logger.error(f"Error parsing predict line: {key} : {e}")
            pred_failed_num += 1
            continue  
    total_wer = 0
    total_ser = 0
    total_words = 0
    micro_wer = jiwer.wer([' '.join(item) for item in true_labels], [' '.join(item) for item in pred_labels])
    print(f"micro_wer: {micro_wer * 100:.2f}")
    for ref, pred in zip(true_labels, pred_labels):
        # 计算当前句子的WER
        current_wer = wer(ref, pred)
        total_wer += current_wer * len(ref)
        total_words += len(ref)
        # 计算当前句子的SER（完全匹配）
        total_ser += 0 if ref == pred else 1
    # 计算平均WER
    avg_wer = total_wer / total_words if total_words > 0 else 0
    # 计算SER
    ser_rate = total_ser / len(true_labels) if len(true_labels) > 0 else 0
    metrics = {
        "WER": round(avg_wer*100, 2),  # 转换为百分比
        "SER": round(ser_rate*100, 2)  # 转换为百分比
    }
    # 使用logger记录结果
    logger.log_prediction_stats(pred_failed_num, total_num)
    logger.info(f"识别结果: {json.dumps(metrics, indent=2)}")

def extract_teaching_feedback(text: str) -> str:
    """
    从文本中提取教学反馈部分
    
    Args:
        text: 包含教学反馈的文本
        
    Returns:
        提取的教学反馈文本
    """
    # 查找教学反馈部分，通常在 "教学反馈：" 之后
    feedback_pattern = r'教学反馈：(.+)'
    match = re.search(feedback_pattern, text)
    
    if match:
        return match.group(1).strip()
    else:
        # 如果没有找到标准格式，尝试其他可能的模式
        # 查找引号内的内容
        quote_pattern = r'["""]([^"""]+)["""]'
        quote_match = re.search(quote_pattern, text)
        if quote_match:
            return quote_match.group(1).strip()
        
        # 如果都没有找到，返回空字符串
        return ""

def extract_word_scores(text: str, logger) -> list:
    """
    从文本中提取单词和对应的分数
    
    Args:
        text: 包含单词分数信息的文本
        
    Returns:
        单词和分数的元组列表
    """
    scores = []
    
    # 查找分数信息，通常在 "***单词发音准确性评测结果：" 之后
    score_pattern = r'\*\*\*单词发音准确性评测结果：\[(.*?)\]\*\*\*'
    score_pattern = r'\*\*\*单词发音准确性评测结果：(\[[^\[\]]*\])\*\*\*'
    match = re.search(score_pattern, text)
    match = re.findall(score_pattern, text)
    if match:
        # score_content = match.group(1)
        score_content = match[0]
        # 解析JSON格式的分数信息
        try:
            # 处理可能的转义字符
            score_content = score_content.replace('\\"', '"')
            # score_data = json.loads(f"[{score_content}]")
            score_data = json.loads(score_content)

            for item in score_data:
                try:
                    word = item['word']
                    score = float(item['score'])
                    score = score_check(dataset_name='open', score_level='word', score_type='accuracy', score=score, logger=logger, data_key=word)
                    scores.append((word, score))
                except (ValueError, TypeError):
                    logger.error(f"Error ValueError ot TypeError: {item}")
                    continue
        except json.JSONDecodeError:
            logger.error(f"Error json.JSONDecodeError, try to parse the score_content.")
            # 如果JSON解析失败，尝试正则表达式提取
            word_score_pattern = r'"word":\s*"([^"]+)"[^}]*"score":\s*"?([0-9.]+)"?'
            matches = re.findall(word_score_pattern, score_content)
            for word, score_str in matches:
                try:
                    score = float(score_str)
                    scores.append((word, score))
                except ValueError:
                    continue
    
    return scores

def load_url_mapping(url_file: str) -> dict:
    """
    从URL文件中加载音频文件名到URL的映射
    
    Args:
        url_file: URL文件路径
        
    Returns:
        音频文件名到URL的映射字典
    """
    url_mapping = {}
    
    if not os.path.exists(url_file):
        print(f"警告：URL文件不存在: {url_file}")
        return url_mapping
    
    try:
        with open(url_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        audio_name = parts[0]
                        url = parts[1]
                        url_mapping[audio_name] = url
    except Exception as e:
        print(f"警告：读取URL文件失败: {e}")
    
    print(f"成功加载 {len(url_mapping)} 个URL映射")
    return url_mapping

def get_open_pa_res(infer_infos, dataset_name: str, logger, output_csv_file: str = None, url_file: str = None):
    """
    处理 Open PA 推理结果
    
    Args:
        infer_infos: [total_num, infer_ress] 格式的推理数据
        dataset_name: 数据集名称
        logger: 日志记录器
        output_csv_file: 输出CSV文件路径（可选）
        url_file: URL文件路径（可选）
    """
    total_num, infer_ress = infer_infos
    pred_failed_num = 0
    data_list = []
    all_predict_scores, all_label_scores = [], []
    all_predict_scores_high, all_label_scores_high = [], []

    # 加载URL映射
    url_mapping = {}
    if url_file and os.path.exists(url_file):
        url_mapping = load_url_mapping(url_file)
    
    for line_num, line in enumerate(infer_ress, 1):
        try:
            line = json.loads(line)
            predict, label = line['predict'], line['label']
            key, ext = os.path.splitext(os.path.basename(line['audio']))            
            # 提取教学反馈
            predict_feedback = extract_teaching_feedback(predict)
            label_feedback = extract_teaching_feedback(label)
            
            # 获取对应的URL
            url = url_mapping.get(key, '')
            
            # 提取分数信息
            predict_scores = extract_word_scores(predict, logger)
            if not predict_scores:
                pred_failed_num += 1
                continue
            label_scores = extract_word_scores(label, logger)

            # 合并单词分为句子分
            if 'open' in dataset_name and 'ket' in dataset_name:
                mean_score = np.nanmean([float(score) for _, score in predict_scores])
                mean_score = score_check('tal-k12', 'sentence', 'accuracy', mean_score, logger, key)
                mean_score = round(mean_score, 0)
                predict_scores = [('nan', mean_score)]

                ref_text = re.findall(r'参考文本：\*\*(.*)\*\*\.', line['prompt'])[0]
                if ref_text == 'none':
                    predict_scores = [('nan', 0)]
                    label_scores = [('nan', 0)]


            if len(predict_scores) != len(label_scores):
                logger.warning(f"{key} 预测分数和标签分数数量不一致: {len(predict_scores)} vs {len(label_scores)}")
                pred_failed_num += 1
                continue

            # 收集所有分数用于计算PCC
            for _, score in predict_scores:
                all_predict_scores.append(score)
            for _, score in label_scores:
                all_label_scores.append(score)
            
            all_above_7 = all(score > 7 for _, score in label_scores)
            if all_above_7:
                all_predict_scores_high.extend([score for _, score in predict_scores])
                all_label_scores_high.extend([score for _, score in label_scores])
            
            # 添加到数据列表（如果需要输出CSV）
            if output_csv_file:
                data_list.append({
                    'audio_name': key,
                    'word_scores_llm': ' '.join([f'{score}' for _, score in predict_scores]),
                    'predict_feedback': predict_feedback,
                    'label_feedback': label_feedback,
                    'url': url
                })
                
        except Exception as e:
            logger.error(f"第 {line_num} 行处理失败: {e}")
            pred_failed_num += 1
            continue
    
    # 使用logger记录结果
    logger.log_prediction_stats(pred_failed_num, total_num)
    
    if all_predict_scores and all_label_scores:
        pcc_compute(all_label_scores, all_predict_scores, dataset_name, logger=logger)
        mse(all_label_scores, all_predict_scores, dataset_name, logger=logger)
        acc_with_threshold(all_label_scores, all_predict_scores, dataset_name=dataset_name, threshold=1, logger=logger)
        
        logger.info(f"预测分数数量: {len(all_predict_scores)}")
        logger.info(f"标签分数数量: {len(all_label_scores)}")
    # 高分pcc
    if all_label_scores_high:
        pcc_compute(all_label_scores_high, all_predict_scores_high, dataset_name, logger=logger)

def get_full_pa_res(infer_infos, dataset_name: str, logger):
    total_num, infer_ress = infer_infos
    pred_failed_num = 0

    all_phn_pre_scores, all_phn_true_scores = [], []
    all_word_prd_scores, all_word_true_scores = [], []
    all_snt_acc_prd_scores, all_snt_acc_true_scores = [], []
    all_snt_flu_prd_scores, all_snt_flu_true_scores = [], []

    for line in infer_ress:
        try:
            line = json.loads(line)
            predict, label = line['predict'], line['label']
            key, ext = os.path.splitext(os.path.basename(line['audio']))

            # 提取音素评测结果
            phn_pre = re.findall(r'音素评测结果：(\[[^\[\]]*\])。', predict)[0]
            phn_pre = json.loads(phn_pre)
            phn_pre_scores = []
            for item in phn_pre:
                tmp_scores = [float(i) for i in item['score'].split()]
                phn_pre_scores.extend(tmp_scores)
            
            phn_true = re.findall(r'音素评测结果：(\[[^\[\]]*\])。', label)[0]
            phn_true = json.loads(phn_true)
            phn_true_scores = []
            for item in phn_true:
                tmp_scores = [float(i) for i in item['score'].split()]
                phn_true_scores.extend(tmp_scores)
            if len(phn_pre_scores) != len(phn_true_scores):
                logger.warning(f"{key} 音素评测结果数量不一致: {len(phn_pre_scores)} vs {len(phn_true_scores)}")
                pred_failed_num += 1
                continue
            for i in range(len(phn_pre_scores)):
                phn_pre_scores[i] = score_check('tal-k12', 'phoneme', 'accuracy', phn_pre_scores[i], logger, key)
                phn_true_scores[i] = score_check('tal-k12', 'phoneme', 'accuracy', phn_true_scores[i], logger, key)
            all_phn_pre_scores.extend(phn_pre_scores)
            all_phn_true_scores.extend(phn_true_scores)

            # 提取单词评测结果
            word_pre = re.findall(r'单词评测结果：(\[.*?\])。', predict)[0]
            word_true = re.findall(r'单词评测结果：(\[.*?\])。', label)[0]
            # word_pre = json.loads(word_pre)
            # word_true = json.loads(word_true)
            # word_pre_scores = [item['score'] for item in word_pre]
            # word_true_scores = [item['score'] for item in word_true]
            word_pre_scores = ast.literal_eval(word_pre)
            word_true_scores = ast.literal_eval(word_true)
            if not word_pre_scores:
                logger.warning(f"{key} 单词评测结果为空，全部设置为0")
                word_pre_scores = [0] * len(word_true_scores)

            if len(word_pre_scores) != len(word_true_scores):
                logger.warning(f"{key} 单词评测结果数量不一致: {len(word_pre_scores)} vs {len(word_true_scores)}")
                pred_failed_num += 1
                continue
            for i in range(len(word_pre_scores)):
                word_pre_scores[i] = score_check('tal-k12', 'word', 'accuracy', word_pre_scores[i], logger, key)
                word_true_scores[i] = score_check('tal-k12', 'word', 'accuracy', word_true_scores[i], logger, key)
            # all_word_prd_scores.extend(word_pre_scores)
            # all_word_true_scores.extend(word_true_scores)

            # 提取句子评测结果
            ## 句子发音准确性
            snt_acc_pre = re.findall(r'句子准确度评测结果：([\d\.]+)，', predict)[0]
            snt_acc_true = re.findall(r'句子准确度评测结果：([\d\.]+)，', label)[0]
            snt_acc_pre = score_check('tal-k12', 'sentence', 'accuracy', snt_acc_pre, logger, key)
            snt_acc_true = score_check('tal-k12', 'sentence', 'accuracy', snt_acc_true, logger, key)
            all_snt_acc_prd_scores.append(snt_acc_pre)
            all_snt_acc_true_scores.append(snt_acc_true)
            ## 句子流畅度
            snt_flu_pre = re.findall(r'句子流利度评测结果：([\d\.]+)。', predict)[0]
            snt_flu_true = re.findall(r'句子流利度评测结果：([\d\.]+)。', label)[0]
            snt_flu_pre = score_check('tal-k12', 'sentence', 'fluency', snt_flu_pre, logger, key)
            snt_flu_true = score_check('tal-k12', 'sentence', 'fluency', snt_flu_true, logger, key)
            all_snt_flu_prd_scores.append(snt_flu_pre)
            all_snt_flu_true_scores.append(snt_flu_true)

            # 临时修正真实标签，测试
            if snt_acc_pre == 0 and snt_flu_pre == 0:
                logger.warning(f"{key} 句子评测结果为0，将单词全部设置为0")
                logger.info(f"{word_pre_scores}")
                logger.info(f"{word_true_scores}")
                logger.info(f"{line['audio'][0]}")
                word_true_scores = [0] * len(word_pre_scores)
            all_word_prd_scores.extend(word_pre_scores)
            all_word_true_scores.extend(word_true_scores)

        except Exception as e:
            logger.error(f"{key} 句子评测结果处理失败: {e}")
            pred_failed_num += 1
            continue
    logger.info(f"预测分数数量: {len(all_word_prd_scores)}")
    logger.info(f"标签分数数量: {len(all_word_true_scores)}")
    logger.info(f"预测句子发音准确性分数数量: {len(all_snt_acc_prd_scores)}")
    logger.info(f"标签句子发音准确性分数数量: {len(all_snt_acc_true_scores)}")
    logger.info(f"预测句子流畅度分数数量: {len(all_snt_flu_prd_scores)}")
    logger.info(f"标签句子流畅度分数数量: {len(all_snt_flu_true_scores)}")
    # 计算PCC
    pcc_compute(all_phn_true_scores, all_phn_pre_scores, f'{dataset_name}_phn_acc', logger=logger)
    pcc_compute(all_word_true_scores, all_word_prd_scores, f'{dataset_name}_word_acc', logger=logger)
    pcc_compute(all_snt_acc_true_scores, all_snt_acc_prd_scores, f'{dataset_name}_snt_acc', logger=logger)
    pcc_compute(all_snt_flu_true_scores, all_snt_flu_prd_scores, f'{dataset_name}_snt_flu', logger=logger)

def get_ket_pa_res(infer_infos, dataset_name: str, logger):
    gt_asr_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/ket/data/test/labels/asr_label_v1105'
    gt_asr = defaultdict(list)
    fo = open('/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v2.1_infer/infer_res/pamllm_pa_score.txt', 'w', encoding='utf-8')
    with open(gt_asr_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            key, asr, answer = line[0], line[2], line[3]
            asr = '' if asr == '<None>' else asr
            answer = '' if answer == '<None>' else answer
            gt_asr[key] = [asr, answer]
    
    doubao_app_asr_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/ket/data/test/asr_res/doubaoapp_asr.txt'
    doubao_app_asr_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/ket/data/test/asr_res/doubao_tal_api_asr.txt'
    huiliu_asr = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/ket/data/test/asr_res/huiliu_asr.txt'
    with open(doubao_app_asr_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            key, asr = line[0], line[1]
            if key in gt_asr:
                gt_asr[key].append(asr)
    with open(huiliu_asr, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            key, asr = line[0], line[1]
            if key in gt_asr:
                gt_asr[key].append(asr)

    total_num, infer_ress = infer_infos
    pred_failed_num = 0
    snt_acc_pred_list, snt_acc_true_list = [], []
    all_asr_wer, all_answer_wer = [], []

    snt_acc_pattern = r'评分为：([\d\.]+)分'
    for line in infer_ress:
        line = json.loads(line)
        key, _ = os.path.splitext(os.path.basename(line['audio']))
        label, predict = line['label'], line['predict']
        try:
            asr_label = re.findall(r'识别结果为：(.*)有效回答文本为', label)[0]
            answer_label = re.findall(r'有效回答文本为：(.*)句子发音准确度评分为', label)[0]
            asr_label = gt_asr[key][0]
            answer_label = gt_asr[key][1]

            asr_pred = re.findall(r'识别结果为：(.*)有效回答文本为', predict)[0]
            answer_pred = re.findall(r'有效回答文本为：(.*)句子发音准确度评分为', predict)[0]

            # answer_pred = text_normalize(answer_pred)
            # asr_pred = text_normalize(asr_pred)
            # fo.write(f"{key}\t{' '.join(asr_pred)}\t{' '.join(answer_pred)}\n")
        except Exception as e:
            pred_failed_num += 1
            logger.error(f"{key} 提取ASR文本失败: {e}")
            # sys.exit(1)
            continue
        
        asr_wer = wer(asr_label, asr_pred)
        answer_wer = wer(answer_label, answer_pred)
        all_asr_wer.append(asr_wer)
        all_answer_wer.append(answer_wer)

        snt_acc_true = re.findall(snt_acc_pattern, label)[0]
        snt_acc_pred = re.findall(snt_acc_pattern, predict)[0]
        snt_acc_pred = score_check('tal-k12', 'sentence', 'accuracy', snt_acc_pred, logger, key)
        if abs(float(snt_acc_pred) - float(snt_acc_true)) > 3:
            logger.warning(f"{key} 句子发音准确性分数差异过大: {snt_acc_pred} vs {snt_acc_true}")
            # continue

        snt_acc_pred_list.append(snt_acc_pred)
        snt_acc_true_list.append(snt_acc_true)
        fo.write(f"{key}\t{snt_acc_pred}\n")
    
    logger.info(f"预测失败数量: {pred_failed_num}/{total_num}")
    logger.info(f"预测句子发音准确性分数数量: {len(snt_acc_pred_list)}")
    logger.info(f"标签句子发音准确性分数数量: {len(snt_acc_true_list)}")
    pcc_compute(snt_acc_true_list, snt_acc_pred_list, f'{dataset_name}_snt_acc', logger=logger)   

    logger.info(f"整体WER: {np.mean(all_asr_wer) * 100:.2f}%")
    logger.info(f"有效回答WER: {np.mean(all_answer_wer) * 100:.2f}%")