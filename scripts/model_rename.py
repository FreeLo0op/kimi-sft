#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import shutil
from safetensors.torch import load_file, save_file

def rename_student_to_model(index_json_path, model_dir, output_dir):
	"""
	1. 删除原始的 model.layers、model.norm、lm_head 权重
	2. 将 student_layers、student_norm、student_head 重命名为 model.layers、model.norm、lm_head
	3. 保存新权重和 index.json 到输出目录
	"""
	# 读取 index.json
	with open(index_json_path, 'r') as f:
		index_data = json.load(f)
	weight_map = index_data["weight_map"]

	# 需要删除的 key 前缀
	remove_prefixes = ["model.layers", "model.norm", "lm_head", "sent_head", "sent_mlp"]
	# 需要重命名的 student 前缀
	student_map = {
		"model.student_layers.": "model.layers.",
		"model.student_norm.": "model.norm.",
		"student_head.": "lm_head."
	}

	# 新的 weight_map
	new_weight_map = {}
	for k, v in weight_map.items():
		# 跳过原始 model 层
		if any(k.startswith(prefix) for prefix in remove_prefixes):
			continue
		# student 层重命名
		for s_prefix, m_prefix in student_map.items():
			if k.startswith(s_prefix):
				new_key = m_prefix + k[len(s_prefix):]
				new_weight_map[new_key] = v
				break
		else:
			# 其它权重保留
			new_weight_map[k] = v

	# 统计需要的权重文件
	needed_files = set(new_weight_map.values())
	# 合并 safetensors 文件
	tensors = {}
	for fname in needed_files:
		fpath = os.path.join(model_dir, fname)
		tensors.update(load_file(fpath))

	# 重新映射 key
	tensors_renamed = {}
	for k in new_weight_map:
		orig_file = new_weight_map[k]
		# 找到 student 原始 key
		for s_prefix, m_prefix in student_map.items():
			if k.startswith(m_prefix):
				student_key = s_prefix + k[len(m_prefix):]
				if student_key in tensors:
					tensors_renamed[k] = tensors[student_key]
					break
		else:
			if k in tensors:
				tensors_renamed[k] = tensors[k]

	# 输出目录
	os.makedirs(output_dir, exist_ok=True)
	# 按原分片保存
	# 统计每个分片需要保存哪些 key
	file_keys = {}
	for k, v in new_weight_map.items():
		file_keys.setdefault(v, []).append(k)
	for fname, keys in file_keys.items():
		out_path = os.path.join(output_dir, fname)
		save_file({k: tensors_renamed[k] for k in keys}, out_path)

	# 保存新的 index.json
	new_index = index_data.copy()
	new_index["weight_map"] = new_weight_map
	out_index_path = os.path.join(output_dir, os.path.basename(index_json_path))
	with open(out_index_path, 'w') as f:
		json.dump(new_index, f, indent=2)
	
	

	print(f"重命名和裁剪完成，输出目录：{output_dir}")

if __name__ == "__main__":
	# 示例用法
    index_json_path = "//mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v3.1/pt_model_distill/checkpoint-7185/model.safetensors.index.json"
    model_dir = "/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v3.1/pt_model_distill/checkpoint-7185"
    output_dir = "/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v3.1/pt_model_distill/checkpoint-7185_renamed"

    rename_student_to_model(index_json_path, model_dir, output_dir)