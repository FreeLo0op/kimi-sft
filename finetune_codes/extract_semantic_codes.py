import os
import sys
import json
import torch.distributed as dist
import argparse
from tqdm import tqdm
from transformers import AutoConfig
from accelerate import Accelerator
from accelerate.utils import gather_object
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
kimia_infer_path = os.path.join(project_root, 'kimia_infer')
if os.path.exists(kimia_infer_path):
    sys.path.insert(0, kimia_infer_path)

from kimia_infer.api.prompt_manager import KimiAPromptManager

def data_loader(input_file:str):
    '''
    逐行读取json文件，返回数据项列表
    '''
    data_items = []
    with open(input_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    logger.info(f"Total {len(lines)} lines in {input_file}")
    for line_idx, line in enumerate(lines):
        try:
            data = json.loads(line.strip())
            data_items.append((data, line_idx))
        except Exception as e:
            logger.error(f"Error parsing line {line_idx} in {input_file}: {e}")
    return data_items

def audio_processor(data_items, prompt_manager, accelerator, batch_size=32):
    '''
    使用accelerate处理数据项，提取语义代码
    '''
    results = []
    processed_audio_count = 0

    # collect all audio data
    audio_tasks = []
    for data, line_idx in data_items:
        for msg in data['conversation']:
            if msg['message_type'] == 'audio':
                audio_tasks.append((msg['content'], msg, data, line_idx))
    if not audio_tasks:
        return [], 0
    
    # process audio in batches
    for i in range(0, len(audio_tasks), batch_size):
        batch = audio_tasks[i:i+batch_size]
        wav_paths = [item[0] for item in batch]
        try:
            wav_tokens_list = prompt_manager._tokenize_audio_batch(wav_paths, batch_size)
            for tokens, (_, msg, data, line_idx) in zip(wav_tokens_list, batch):
                msg['audio_tokens'] = tokens
                results.append((json.dumps(data, ensure_ascii=False) + '\n', line_idx))
                processed_audio_count += 1
        except Exception as e:
            print(f"Error processing audio batch starting at index {i}: {e}")
            pass

    return results, processed_audio_count

def main(args):

    accelerator = Accelerator()
    accelerator.print(f"使用 {accelerator.num_processes} 个进程")
    accelerator.print(f"当前进程: {accelerator.process_index}/{accelerator.num_processes}")
    accelerator.print(f"设备: {accelerator.device}")

    # load model config
    model_config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # load data
    data_items = data_loader(args.input_file)

    # initialize prompt manager
    accelerator.print("初始化模型...")
    prompt_manager = KimiAPromptManager(
        model_path=args.model_name_or_path,
        audio_tokenizer=args.audio_tokenizer,
        kimia_token_offset=model_config.kimia_token_offset, 
        kimia_text_audiodelaytokens=model_config.kimia_mimo_audiodelaytokens,
        device=accelerator.device
    )
    accelerator.print("模型初始化完成")

    # 数据切片
    data_items = data_items[accelerator.process_index::accelerator.num_processes]
    accelerator.print(f"进程 {accelerator.process_index} 处理 {len(data_items)} 个数据项")

    # 批量处理
    all_results = []
    processed_audio_count = 0
    bar = tqdm(total=len(data_items), desc=f"Process {accelerator.process_index}") if accelerator.is_main_process else None
    for i in range(0, len(data_items), args.batch_size):
        batch = data_items[i:i + args.batch_size]
        results, batch_audio_count = audio_processor(
            batch, 
            prompt_manager, 
            accelerator, 
            batch_size=args.batch_size
        )
        all_results.extend(results)
        processed_audio_count += batch_audio_count
        if bar:
            bar.update(len(batch))
    if bar:
        bar.close()

    gathered_results = gather_object(all_results)
    if accelerator.is_main_process:
        accelerator.print("开始写入结果...")
        gathered_results.sort(key=lambda x: x[1])  # 按行号排序
        with open(args.output_file, 'w', encoding='utf-8') as fo:
            for json_line, line_idx in gathered_results:
                fo.write(json_line)
        accelerator.print(f"写入完成，处理了 {len(gathered_results)} 行数据。")
    accelerator.wait_for_everyone()
    accelerator.print("所有数据处理完成！")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/Kimi-Audio-7B")
    parser.add_argument("--audio_tokenizer", type=str, default="/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/THUDM/glm-4-voice-tokenizer")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32, help="批量处理大小")
    args = parser.parse_args()

    main(args)
