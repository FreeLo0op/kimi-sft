import logging
import os
from datetime import datetime
from typing import Optional

SOTA = {
    '4377_align_cotv2_test': 0.83,
    'next1_align_cotv2_test':0.84,
    'next2_align_cotv2_test':0.85,
    'align_cn_test_align_cotv1_test': 0.85,
    'speechocean762_phoneme_pa_cotv2_test': 0.72,
    'tal-k12_phoneme_pa_nocot-v1_test': 0.73,
    'speechocean762_word_pa_accuracy_nocot-v2_test': 0.70,
    'tal-k12_word_pa_accuracy_nocot-v2_test': 0.77,
    'speechocean762_word_pa_stress_nocot-v2_test': 0.42,
    'speechocean762_word_pa_total_nocot-v2_test': 0.73,

    'speechocean762_sent_pa_accuracy_nocot-v2_test': 0.78,
    'tal-k12_sent_pa_accuracy_nocot-v2_test': 0.94,
    'speechocean762_sent_pa_fluency_nocotv1_test': 0.84,
    'next_sent_pa_fluency_nocotv1_test': 0.78,
    'speechocean762_sent_pa_prosodic_nocot-v2_test': 0.84,
    'speechocean762_sent_pa_total_nocot-v2_test': 0.82,

    'sent_acc_test_sent_pa_accuracy_nocotv2_test': 0.81
}

class EvaluationLogger:
    """评估结果日志记录器"""
    
    def __init__(self, log_file: Optional[str] = None, level: int = logging.INFO, clear_existing: bool = True):
        """
        初始化日志记录器
        
        Args:
            log_file: 日志文件路径，如果为None则只输出到控制台
            level: 日志级别
            clear_existing: 是否清理已存在的日志文件，默认为True
        """
        self.logger = logging.getLogger('evaluation')
        self.logger.setLevel(level)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] >>> %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 如果指定了日志文件，添加文件处理器
        if log_file:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # 如果日志文件已存在且需要清理，则删除现有文件
            if clear_existing and os.path.exists(log_file):
                try:
                    os.remove(log_file)
                    print(f"已清理现有日志文件: {log_file}")
                except Exception as e:
                    print(f"清理日志文件失败: {e}")
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """记录信息级别的日志"""
        self.logger.info(message, stacklevel=2)
    
    def warning(self, message: str):
        """记录警告级别的日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误级别的日志"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """记录调试级别的日志"""
        self.logger.debug(message)
    
    def log_dataset_info(self, dataset_name: str, dataset_key: str, length: int):
        """记录数据集信息"""
        self.info(f"Dataset: {dataset_key}({dataset_name}), Length: {length}")
        if SOTA.get(dataset_key):
            self.info(f"SOTA: {SOTA.get(dataset_key)}")
    
    def log_prediction_stats(self, pred_failed_num: int, total_num: int, diff_num: int = 0):
        """记录预测统计信息"""
        self.info(f"预测失败数量: {pred_failed_num} / {total_num} ({pred_failed_num/total_num:.2%})")
        if diff_num > 0:
            self.info(f"预测和标签不一致数量: {diff_num} / {total_num} ({diff_num/total_num:.2%})")
    
    def log_metric_result(self, metric_name: str, value: float, dataset_name: str = ""):
        """记录评估指标结果"""
        if dataset_name:
            self.info(f"[{dataset_name}] {metric_name}: {value:.4f}")
        else:
            self.info(f"{metric_name}: {value:.4f}")
    
    def log_separator(self):
        """记录分隔线"""
        self.info("-" * 50 + '\n')

# 全局日志记录器实例
_global_logger = None

def get_logger() -> EvaluationLogger:
    """获取全局日志记录器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = EvaluationLogger()
    return _global_logger

def set_logger(log_file: Optional[str] = None, level: int = logging.INFO, clear_existing: bool = True):
    """
    设置全局日志记录器
    
    Args:
        log_file: 日志文件路径，如果为None则只输出到控制台
        level: 日志级别
        clear_existing: 是否清理已存在的日志文件，默认为True
    """
    global _global_logger
    _global_logger = EvaluationLogger(log_file, level, clear_existing)
    return _global_logger 