#!/bin/bash

# 音频语义编码提取脚本
# 使用accelerate进行多GPU加速

# 检查accelerate是否安装
if ! command -v accelerate &> /dev/null; then
    echo "错误: accelerate未安装，请先安装:"
    echo "pip install accelerate"
    exit 1
fi

# 设置参数
# INPUT_FILE=${1:-"input.json"}
# OUTPUT_FILE=${2:-"output.json"}
BATCH_SIZE=32
NUM_PROCESSES=8

GPU_IDS=${5:-"0,1,2,3,4,5,6,7"}  # 默认使用GPU 0-7

if [ "$OUTPUT_FILE" == "output.json" ]; then
    echo "使用默认输出路径"
    OUTPUT_FILE="${INPUT_FILE%.*}_semantic_codes.json"
fi

echo "开始音频语义编码提取..."
echo "输入文件: $INPUT_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "批量大小: $BATCH_SIZE"
echo "进程数: $NUM_PROCESSES"
echo "使用GPU: $GPU_IDS"
echo "================================"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# /mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/CPT_v1_Stage1/train/spilt_one_data
INPUT_FILES=(
    '/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/CPT_v1_Stage1/train/spilt_one_data/split_aa'
    '/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/CPT_v1_Stage1/train/spilt_one_data/split_ab'
    # '/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/CPT_v1_Stage1/train/spilt_one_data/split_ac'
    # '/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/CPT_v1_Stage1/train/spilt_one_data/split_ad'
    # '/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/CPT_v1_Stage1/train/spilt_one_data/split_ae'
    # '/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/CPT_v1_Stage1/train/spilt_one_data/split_af'
    # '/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/CPT_v1_Stage1/train/spilt_one_data/split_ag'
    # '/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/CPT_v1_Stage1/train/spilt_one_data/split_ah'
    # '/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/CPT_v1_Stage1/train/spilt_one_data/split_ai'
    # '/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/CPT_v1_Stage1/train/spilt_one_data/split_aj'
)
for INPUT_FILE in "${INPUT_FILES[@]}"; do
    OUTPUT_FILE="${INPUT_FILE%.*}_semantic_codes.json"
    echo "Processing $INPUT_FILE ... to $OUTPUT_FILE"
    if [ ! -f "$INPUT_FILE" ]; then
        echo "输入文件 $INPUT_FILE 不存在，跳过..."
        continue
    fi
    # 使用accelerate启动
    accelerate launch \
        --config_file accelerate_config.yaml \
        --num_processes $NUM_PROCESSES \
        extract_semantic_codes.py \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE" \
        --batch_size $BATCH_SIZE

    echo "Finished processing $INPUT_FILE, output saved to $OUTPUT_FILE"
    echo "--------------------------------"
done
# 使用accelerate启动
# accelerate launch \
#     --config_file accelerate_config.yaml \
#     --num_processes $NUM_PROCESSES \
#     extract_semantic_codes.py \
#     --input_file "$INPUT_FILE" \
#     --output_file "$OUTPUT_FILE" \
#     --batch_size $BATCH_SIZE

# echo "处理完成！"