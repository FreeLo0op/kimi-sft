#!/usr/bin/env python3
"""
merge_extra_head.py

把模型 A 中以指定前缀（默认 "extra_head"）开头的权重复制到模型 B 并保存为新的模型文件。

支持的输入格式：
- 单文件 safetensors (.safetensors)
- safetensors 索引文件 (.index.json + 对应 shards)
- PyTorch 保存的 state_dict (.pt/.pth 或 pytorch_model.bin)

输出：可以选择保存为 .pt 或 单文件 .safetensors

示例：
  python scripts/merge_extra_head.py \
    --model_a /path/to/modelA/model.safetensors.index.json \
    --model_b /path/to/modelB/pytorch_model.bin \
    --out merged_with_extra_head.pt \
    --prefix extra_head

依赖：torch, safetensors (用于 safetensors 格式)
    pip install torch safetensors
"""

import os
import sys
import json
import argparse
from typing import Dict

try:
    import torch
except Exception:
    print("错误：请先安装 torch（pip install torch）", file=sys.stderr)
    raise

try:
    from safetensors.torch import load_file as safetensors_load, save_file as safetensors_save
except Exception:
    safetensors_load = None
    safetensors_save = None


def load_from_index(index_path: str) -> Dict[str, "torch.Tensor"]:
    with open(index_path, 'r', encoding='utf-8') as f:
        idx = json.load(f)

    # 常见格式：{ "weight_map": { key: shard_filename, ... } }
    weight_map = idx.get('weight_map') or idx.get('model_chunks') or idx.get('weightmap')
    if not weight_map:
        raise ValueError(f"在 {index_path} 中找不到 weight_map 字段，无法解析索引文件。")

    base = os.path.dirname(index_path)
    state = {}

    # 按 shard 文件批量载入以减少重复读取
    shards = {}
    for k, v in weight_map.items():
        shards.setdefault(v, []).append(k)

    if safetensors_load is None:
        raise RuntimeError("未安装 safetensors：请运行 pip install safetensors")

    for shard_file, keys in shards.items():
        shard_path = os.path.join(base, shard_file)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"找不到 shard 文件: {shard_path}")
        shard_dict = safetensors_load(shard_path)
        for k in keys:
            if k not in shard_dict:
                raise KeyError(f"在 {shard_path} 中未找到 key {k}")
            state[k] = shard_dict[k].cpu()

    return state


def load_state_dict(path: str) -> Dict[str, "torch.Tensor"]:
    # 支持传入目录（尝试在目录中寻找常见文件名）
    if os.path.isdir(path):
        candidates = [
            'model.safetensors.index.json',
            'pytorch_model.bin.index.json',
            'model.safetensors',
            'pytorch_model.bin',
            'pytorch_model.pt',
        ]
        for c in candidates:
            p = os.path.join(path, c)
            if os.path.exists(p):
                return load_state_dict(p)
        raise FileNotFoundError(f"目录 {path} 中未找到已知模型文件，请给出具体文件路径。")

    path = os.path.abspath(path)
    if path.endswith('.index.json'):
        return load_from_index(path)

    if path.endswith('.safetensors'):
        if safetensors_load is None:
            raise RuntimeError("未安装 safetensors：请运行 pip install safetensors")
        return safetensors_load(path)

    # PyTorch 格式
    try:
        return torch.load(path, map_location='cpu')
    except Exception as e:
        raise ValueError(f"无法加载文件 {path}：{e}")


def save_state_dict(state: Dict[str, "torch.Tensor"], out_path: str):
    out_path = os.path.abspath(out_path)
    if out_path.endswith('.safetensors'):
        if safetensors_save is None:
            raise RuntimeError("未安装 safetensors：请运行 pip install safetensors")
        # 确保 tensor 在 cpu
        cpu_state = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in state.items()}
        safetensors_save(cpu_state, out_path)
    else:
        # 默认保存为 torch 的 pt
        torch.save(state, out_path)


def main():
    model_a = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_test/history_v1/model_infer'
    model_b = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v2.2/model_infer_merge'
    save_out = f'{model_b}/extra_head.safetensors'

    print(f"载入模型 A: {model_a}")
    state_a = load_state_dict(model_a)
    print(f"模型 A 包含 {len(state_a)} 个参数 key")

    print(f"载入模型 B: {model_b}")
    state_b = load_state_dict(model_b)
    print(f"模型 B 包含 {len(state_b)} 个参数 key")

    # 选择要复制的 keys
    keys_to_copy = [k for k in state_a.keys() if k.startswith('extra_head')]
    if not keys_to_copy:
        print(f"未在模型 A 中找到以 'extra_head' 开头的参数。请确认前缀是否正确。")
        sys.exit(1)

    print(f"找到 {len(keys_to_copy)} 个将被复制的参数（前缀='extra_head'）")
    for k in keys_to_copy:
        print('  ' + k)

    # return
    # 复制到 B
    state_extra_head = {}
    for k in keys_to_copy:
        state_extra_head[k] = state_a[k]

    save_state_dict(state_extra_head, save_out)
    print("保存完成。建议在推理前根据目标框架/代码加载并校验模型（或保留备份）。")

if __name__ == '__main__':
    main()
