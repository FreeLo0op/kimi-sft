import os
import sys
from postprocess.infer_res_pp import *
from logger import get_logger, set_logger
import logging

def main(
    infer_res_dir:str,
    infer_config_file:str=None,
    log_file:str=None,
    clear_log:bool=True,
):
    # 设置日志记录器 - 如果log_file为None，则只输出到控制台
    log_file = os.path.join(infer_res_dir, log_file) if log_file is not None else 'evaluation.log'
    logger = set_logger(log_file, clear_existing=clear_log)
    
    dataset_info_dict = defaultdict(list)
    for file in os.listdir(infer_res_dir):
        dataset_name = file.replace('infer_', '').replace('.json', '')
        file = os.path.join(infer_res_dir, file)
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [len(lines), lines]
        dataset_info_dict[dataset_name] = lines

    for key, values in dataset_info_dict.items():
        dataset_name = key.split('_')[0]
        if not values[1]:
            continue
        logger.log_dataset_info(dataset_name, key, values[0])

        if 'align' in key:
            get_align_res(values, logger)
        elif 'phoneme_pa' in key:
            get_phoneme_pa_res(values, dataset_name, logger)
        elif 'word_pa' in key:
            pa_type = next((v for k, v in {
                'accuracy': 'accuracy',
                'stress': 'stress',
                'total': 'total'
            }.items() if k in key), None)
            get_word_pa_res(values, pa_type, dataset_name, logger)
        elif 'sent_pa' in key:
            get_snt_pa_res(values, dataset_name, logger)
        elif 'full_pa' in key:
            get_full_pa_res(values, dataset_name, logger)
        elif 'multi_pa' in key:
            get_snt_multi_pa_res(values, dataset_name, logger)
        elif 'liaison' in key:
            get_liaison_res(values, logger)
        elif 'multiple_choice' in key:
            get_multiple_choice_res(values, logger)
        elif 'repeat_reading' in key:
            get_repeat_reading_res(values, logger)
        elif 'asr' in key:
            get_asr_res(values, logger)
        elif 'full_pa' in key:
            get_full_pa_res(values, dataset_name, logger)
        elif 'ket_pa' in key:
            get_ket_pa_res(values, dataset_name, logger)
        elif 'xxj' in key:
            get_xxj_res(values, dataset_name, logger)
        elif 'ipa' in key:
            get_ipa_res(values, logger)
        else:
            logger.error(f"Unknown task type: {key}")
        logger.log_separator()

def single_main(infer_res_file:str, task_type:str, log_file:str=None, clear_log:bool=True):
    # 设置日志记录器 - 如果log_file为None，则只输出到控制台
    logger = set_logger(log_file, clear_existing=clear_log)
    
    with open(infer_res_file, 'r') as f:
        lines = f.readlines()
    lines = [len(lines), lines]
    match task_type:
        case 'align':
            get_align_res(lines, logger)
        case 'phoneme_pa':
            get_phoneme_pa_res(lines, 'tal-k12', logger)
        case 'word_pa':
            get_word_pa_res(lines, 'accuracy', 'tal-k12', logger)  # Default to 'accuracy' for word_pa
        case 'sent_pa':
            get_snt_pa_res(lines, 'tal-k12', logger)
        case 'liaison':
            get_liaison_res(lines, logger)
        case 'multiple_choice':
            get_multiple_choice_res(lines, logger)
        case 'repeat_reading':
            get_repeat_reading_res(lines, logger)
        case 'asr':
            get_asr_res(lines, logger)
        case 'open_pa':
            get_open_pa_res(lines, 'open_pa', logger)
        case 'full_pa':
            get_full_pa_res(lines, 'full_pa', logger)
        case 'ket_pa':
            get_ket_pa_res(lines, 'ket_pa', logger)
        case 'xxj':
            get_xxj_res(lines, 'xxj', logger)
        case 'ipa':
            get_ipa_res(lines, logger)
        case _:
            logger.error(f"Unknown task type: {task_type}")
    pass

if __name__ == "__main__":

    # main(
    #     infer_res_dir='/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v2.3/model_infer_2/infer_res', 
    #     log_file='evaluation_all.log', 
    #     clear_log=True
    # )

    infer_res_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v2.7/model_infer/infer_res/infer_tal-k12-td1_full_pa_llmgt_cotv1_test.json'
    log_file = os.path.join(os.path.dirname(infer_res_file), 'evaluation.log')
    single_main(
        infer_res_file=infer_res_file,
        task_type='full_pa',
        log_file=log_file,
        clear_log=False
    )