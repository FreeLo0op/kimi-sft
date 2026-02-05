#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=ERROR

# 如果 NCCL 超时短，可以调长：
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_ENABLE_MONITORING=1

DIR=`pwd`

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

while [[ "$1" != "" ]]; do
    case $1 in
        --node_rank )
            shift
            NODE_RANK=$1
            ;;
        -h | --help )
            echo "Usage: bash finetune_ds.sh [--node_rank NUM]"
            echo "  --node_rank NUM: Rank of the current node (0 for master, 1 for worker, etc.)"
            echo "  Default is single-node mode with NODE_RANK=0, NNODES=1, MASTER_ADDR=localhost."
            exit 0
            ;;
        * )
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

if [ -z ${NODE_RANK+x} ]; then
    # 单节点模式
    NODE_RANK=0
    NNODES=1
    MASTER_ADDR=localhost
    MASTER_PORT=6001
else
    # 多节点模式
    NNODES=2
    MASTER_ADDR="10.198.67.221"  # Set the IP address (or hostname) of the master node
    MASTER_PORT=6001
fi

MODEL="/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/Kimi-Audio-7B" # Set the path if you do not want to load from huggingface directly

PRETRAINED_MODEL_PATH="/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/Kimi-Audio-7B"

# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA_TRAIN="/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/CPT_v1_Stage1/train/train_30_semantic_codes.json"
DATA_EVAL="/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/CPT_v1_Stage1/eval/eval_30_semantic_codes.json"
output_dir="/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/Base_Model/Kimi-PA-Base-v3/CPT_STAGE1_MODEL"
batch_size=8
model_max_length=1024

echo "PRETRAINED_MODEL_PATH: $PRETRAINED_MODEL_PATH"
echo "DATA: $DATA_TRAIN"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo "start finetune"
echo "DISTRIBUTED_ARGS: $DISTRIBUTED_ARGS"

cd /mnt/pfs_l2/jieti_team/SFT/hupeng/github/kimi-sft
torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --model_path $PRETRAINED_MODEL_PATH \
    --train_data_path $DATA_TRAIN \
    --eval_data_path $DATA_EVAL \
    --eval_ratio 0.05 \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --model_max_length $model_max_length \
    --gradient_accumulation_steps 8 \
    --eval_strategy "steps" \
    --save_strategy "steps" \
    --eval_steps 3000 \
    --save_steps 3000 \
    --save_total_limit 2000 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed finetune_codes/ds_config_zero3.json \
    --load_audio_head False