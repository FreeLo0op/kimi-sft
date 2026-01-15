#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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
    MASTER_ADDR="10.207.5.82"  # Set the IP address (or hostname) of the master node
    MASTER_PORT=6001
fi

MODEL="/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/Kimi-Audio-7B" # Set the path if you do not want to load from huggingface directly

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
echo "start finetune"
echo "DISTRIBUTED_ARGS: $DISTRIBUTED_ARGS"

# PRETRAINED_MODEL_PATH=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v3.1/pt_model_distill_2/checkpoint-7185_renamed
PRETRAINED_MODEL_PATH=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/Base_Model/Kimi-PA-Base-v2/checkpoint-75000
# DATA_TRAIN=/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/audio_detect/train/audio_detect_train_semantic_codes.json

DATA_TRAIN=/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/sft/train/train_29_semantic_codes.json
DATA_EVAL=/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/sft/dev/eval_29_semantic_codes.json
# DATA_TRAIN=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v3.1/train_25_semantic_codes.json
# DATA_EVAL=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v3.1/eval_25_semantic_codes.json

output_dir=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v3.5/pt_model
# output_dir=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v3.1/pt_model_distill_2_epoch3_sft
batch_size=2
model_max_length=2048

cd /mnt/pfs_l2/jieti_team/SFT/hupeng/github/kimi-sft
torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --model_path $PRETRAINED_MODEL_PATH \
    --train_data_path $DATA_TRAIN \
    --eval_data_path $DATA_EVAL \
    --eval_ratio 0.05 \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps 8 \
    --eval_strategy "steps" \
    --save_strategy "epoch" \
    --eval_steps 400 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --report_to "tensorboard" \
    --model_max_length $model_max_length \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed finetune_codes/ds_config_zero3.json \
    --load_audio_head False

cp /mnt/pfs_l2/jieti_team/SFT/hupeng/github/kimi-sft/finetune_codes/finetune_sft_ds.sh $output_dir/finetune_sft_ds.sh.backup
cp $DATA_EVAL $output_dir/eval.json.backup
echo "Finetune process completed."