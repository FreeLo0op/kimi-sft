#!/bin/bash

# base模型性能测试
# 测试集：1、asr：Librispeech、aishell 
#        2、pa：tal-k12-word-accuracy，tal-k12-sentence-accuracy


# Convert the finetuned model for inference.
convert=$1  # "true" or "false"

if [ "$convert" == "true" ]; then
    echo "Converting model for inference..."
    
    cpkt='55000'
    model_name=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/Kimi-Audio-7B
    input_dir=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/Base_Model/Kimi-PA-Base-v2/checkpoint-${cpkt}
    output_dir=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/Base_Model/Kimi-PA-Base-v2/Kimi_Pa_V2_ckpt${cpkt}

    cd /mnt/pfs_l2/jieti_team/SFT/hupeng/Kimi-Audio
    python -m finetune_codes.model --model_name $model_name \
        --action "export_model" \
        --input_dir $input_dir \
        --output_dir $output_dir
    echo "Model conversion completed."
fi

echo "Skipping model conversion."
cd /mnt/pfs_l2/jieti_team/SFT/hupeng/github/kimi-sft
infer_prompt='/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/base_model_test_dataset/test_dataset.json'
model_path=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/Kimi_test_2_infer
output_path=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/Kimi_test_2_infer/asr_test_results.json
gpu_id=6
CUDA_VISIBLE_DEVICES="$gpu_id" python infer.py \
    --model_path "$model_path" \
    --infer_prompt "$infer_prompt" \
    --output_path "$output_path" \
    --gpu_id "0"
