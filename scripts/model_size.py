#!/usr/bin/env python3
import os
import argparse
import json
from pathlib import Path
from safetensors import safe_open
import torch
import humanize
from collections import defaultdict

def get_file_size(file_path):
    """获取文件大小（人类可读格式）"""
    return humanize.naturalsize(os.path.getsize(file_path))

def categorize_layer(layer_name):
    """根据层名称分类"""
    categories = {
        'whisper_model': 'whisper_model',
        'model.layers': 'model.layers', 
        'model.mimo_layers': 'model.mimo_layers',
        'lm_head': 'lm_head',
        'mimo_output': 'mimo_output'
    }
    
    for category_prefix in categories:
        if layer_name.startswith(category_prefix):
            return categories[category_prefix]
    
    return 'other'

def calculate_layer_sizes(safetensors_path):
    """计算safetensors文件中每层的参数大小"""
    layer_sizes = {}
    total_size_bytes = 0
    
    try:
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # 计算参数数量
                num_params = tensor.numel()
                # 计算内存大小（字节）
                size_bytes = tensor.numel() * tensor.element_size()
                total_size_bytes += size_bytes
                
                layer_sizes[key] = {
                    'shape': list(tensor.shape),
                    'num_params': num_params,
                    'size_bytes': size_bytes,
                    'size_human': humanize.naturalsize(size_bytes),
                    'dtype': str(tensor.dtype),
                    'category': categorize_layer(key)
                }
                
    except Exception as e:
        print(f"Error reading {safetensors_path}: {e}")
        return None, 0
    
    return layer_sizes, total_size_bytes

def analyze_safetensors_directory(directory):
    """分析目录下的所有safetensors文件"""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return
    
    safetensors_files = list(directory.rglob("*.safetensors"))
    
    if not safetensors_files:
        print(f"No .safetensors files found in {directory}")
        return
    
    print(f"Found {len(safetensors_files)} .safetensors files in {directory}")
    print("=" * 80)
    
    # 初始化分类统计
    category_stats = defaultdict(lambda: {
        'total_params': 0,
        'total_size_bytes': 0,
        'layer_count': 0,
        'layers': []
    })
    
    total_files_size = 0
    all_layers = {}
    
    for file_path in safetensors_files:
        print(f"\n📁 Analyzing: {file_path.relative_to(directory)}")
        print(f"📏 File size: {get_file_size(file_path)}")
        
        layer_sizes, total_size_bytes = calculate_layer_sizes(file_path)
        
        if layer_sizes is None:
            continue
            
        total_files_size += total_size_bytes
        all_layers.update(layer_sizes)
        
        # 按类别统计
        for layer_name, info in layer_sizes.items():
            category = info['category']
            category_stats[category]['total_params'] += info['num_params']
            category_stats[category]['total_size_bytes'] += info['size_bytes']
            category_stats[category]['layer_count'] += 1
            category_stats[category]['layers'].append(layer_name)
    
    # 打印分类统计
    print_category_summary(category_stats, total_files_size)
    
    # 打印详细层信息
    print_detailed_layer_info(all_layers)
    
    return all_layers, category_stats

def print_category_summary(category_stats, total_size_bytes):
    """打印按类别分类的统计信息"""
    print("\n" + "=" * 80)
    print("🎯 CATEGORY SUMMARY")
    print("=" * 80)
    
    # 定义类别显示顺序
    category_order = [
        'whisper_model',
        'model.layers', 
        'model.mimo_layers',
        'lm_head',
        'mimo_output',
        'other'
    ]
    
    print(f"\n{'Category':<20} {'Layers':<8} {'Params':<15} {'Size':<15} {'% of Total':<10}")
    print("-" * 80)
    
    total_params = sum(stats['total_params'] for stats in category_stats.values())
    
    for category in category_order:
        if category in category_stats:
            stats = category_stats[category]
            param_pct = (stats['total_params'] / total_params) * 100 if total_params > 0 else 0
            size_pct = (stats['total_size_bytes'] / total_size_bytes) * 100 if total_size_bytes > 0 else 0
            
            print(f"{category:<20} {stats['layer_count']:<8} "
                  f"{humanize.intcomma(stats['total_params']):<15} "
                  f"{humanize.naturalsize(stats['total_size_bytes']):<15} "
                  f"{size_pct:.1f}%")

def print_detailed_layer_info(all_layers):
    """打印每个类别的详细层信息"""
    print("\n" + "=" * 80)
    print("📋 DETAILED LAYER INFORMATION")
    print("=" * 80)
    
    # 按类别分组
    categorized_layers = defaultdict(dict)
    for layer_name, info in all_layers.items():
        categorized_layers[info['category']][layer_name] = info
    
    # 打印每个类别的详细信息
    for category in ['whisper_model', 'model.layers', 'model.mimo_layers', 'lm_head', 'mimo_output', 'other']:
        if category in categorized_layers and categorized_layers[category]:
            print(f"\n🏷️  {category.upper()} Layers:")
            print("-" * 100)
            print(f"{'Layer Name':<60} {'Shape':<20} {'Params':<15} {'Size':<15}")
            print("-" * 100)
            
            # 按大小排序
            sorted_layers = sorted(categorized_layers[category].items(), 
                                 key=lambda x: x[1]['size_bytes'], 
                                 reverse=True)
            
            for layer_name, info in sorted_layers:
                params_str = humanize.intcomma(info['num_params'])
                print(f"{layer_name:<60} {str(info['shape']):<20} {params_str:<15} {info['size_human']:<15}")

def save_to_json(all_layers, category_stats, output_path):
    """将结果保存为JSON文件"""
    output_data = {
        'total_parameters': sum(l['num_params'] for l in all_layers.values()),
        'total_size_bytes': sum(l['size_bytes'] for l in all_layers.values()),
        'category_stats': {},
        'layers': all_layers
    }
    
    for category, stats in category_stats.items():
        output_data['category_stats'][category] = {
            'total_params': stats['total_params'],
            'total_size_bytes': stats['total_size_bytes'],
            'layer_count': stats['layer_count'],
            'layers': stats['layers']
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_path}")

def print_overall_statistics(all_layers, category_stats):
    """打印总体统计信息"""
    total_params = sum(l['num_params'] for l in all_layers.values())
    total_size = sum(l['size_bytes'] for l in all_layers.values())
    
    print("\n" + "=" * 80)
    print("📊 OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total parameters: {humanize.intcomma(total_params)}")
    print(f"Total model size: {humanize.naturalsize(total_size)}")
    print(f"Total layers: {len(all_layers)}")
    
    # 打印每个类别的参数分布
    print(f"\nParameter distribution by category:")
    for category, stats in category_stats.items():
        pct = (stats['total_params'] / total_params) * 100
        print(f"  {category}: {pct:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Analyze .safetensors files and calculate layer sizes by category')
    parser.add_argument('directory', type=str, help='Directory to search for .safetensors files')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print(f"🔍 Searching for .safetensors files in: {args.directory}")
    
    all_layers, category_stats = analyze_safetensors_directory(args.directory)
    
    # 打印总体统计
    if all_layers:
        print_overall_statistics(all_layers, category_stats)
    
    # 保存结果
    if args.output and all_layers:
        save_to_json(all_layers, category_stats, args.output)

if __name__ == "__main__":
    main()