#!/bin/bash

set -euo pipefail
convert=$1  # "true" or "false"
if [ "$convert" == "true" ]; then
    echo "Converting model for inference..."

    model_name=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/Kimi-Audio-7B
    input_dir=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v2.1
    output_dir=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v2.1_infer

    cd /mnt/pfs_l2/jieti_team/SFT/hupeng/Kimi-Audio
    python -m finetune_codes.model --model_name $model_name \
        --action "export_model" \
        --input_dir $input_dir \
        --output_dir $output_dir
    echo "Model conversion completed."
fi
cd /mnt/pfs_l2/jieti_team/SFT/hupeng/github/kimi-sft

testDataset_list=/mnt/pfs_l2/jieti_team/SFT/hupeng/github/kimi-sft/utils/configs/testDataset.list
root_dir='/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/multi_task/sft/test'
output_dir=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v2.1_infer/infer_res
model_path=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v2.1_infer
num_gpus=4

if [[ ! -f "$testDataset_list" ]]; then
	echo "test dataset list not found: $testDataset_list" >&2
	exit 1
fi

mkdir -p "$output_dir"

mapfile -t datasets < <(grep -Ev '^\s*(#|$)' "$testDataset_list")

if [[ ${#datasets[@]} -eq 0 ]]; then
	echo "No datasets to process." >&2
	exit 0
fi

declare -a gpu_pid

launch_job() {
	local gpu_id="$1"
	local dataset="$2"

	local infer_prompt="$root_dir/$dataset"
	if [[ ! -f "$infer_prompt" ]]; then
		echo "[GPU $gpu_id] Skip missing dataset file: $infer_prompt" >&2
		return 1
	fi

	local output_path="$output_dir/infer_${dataset}"

	echo "[GPU $gpu_id] Start: $dataset"
	CUDA_VISIBLE_DEVICES="$gpu_id" python infer.py \
		--model_path "$model_path" \
		--infer_prompt "$infer_prompt" \
		--output_path "$output_path" \
		--gpu_id "0" &
    # sleep 5  # Simulate job duration for testing
	gpu_pid[$gpu_id]=$!
}

for dataset in "${datasets[@]}"; do
	while true; do
		for (( gpu=0; gpu<num_gpus; gpu++ )); do
			pid="${gpu_pid[$gpu]:-}"
			if [[ -z "$pid" ]]; then
				launch_job "$gpu" "$dataset" && break 2
			fi

			if ! kill -0 "$pid" 2>/dev/null; then
				wait "$pid" || echo "[GPU $gpu] Job exited with error." >&2
				unset gpu_pid[$gpu]
				launch_job "$gpu" "$dataset" && break 2
			fi
		done
		sleep 2
	done
done

for pid in "${gpu_pid[@]}"; do
	if [[ -n "$pid" ]]; then
		wait "$pid" || echo "Job with PID $pid failed." >&2
	fi
done

echo "All inference jobs completed."