#!/usr/bin/env python3
"""
解析 grpo.log 文件，提取 score, hallucination, correct, miss 等指标并绘制折线图
"""

import re
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def parse_log_file(log_file_path):
    """解析日志文件，提取各个指标数据"""
    data = defaultdict(lambda: defaultdict(list))
    current_step = None
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 检测 Step 标记
        step_match = re.search(r'Training Sample @ Step (\d+)', line)
        if step_match:
            current_step = int(step_match.group(1))
            print(f"Found Step {current_step}")
        
        # 检测指标字典的开始（包含 'val/test_' 的行）
        if current_step and "'val/test_" in line and '{' in line:
            # 读取完整的字典，可能跨越多行
            dict_lines = [line]
            bracket_count = line.count('{') - line.count('}')
            
            j = i + 1
            while bracket_count > 0 and j < len(lines):
                dict_lines.append(lines[j])
                bracket_count += lines[j].count('{') - lines[j].count('}')
                j += 1
            
            dict_str = ''.join(dict_lines)
            
            # 去除 ANSI 颜色代码
            dict_str = re.sub(r'\[36m.*?\[0m', '', dict_str)
            dict_str = re.sub(r'\(main_task pid=\d+\)', '', dict_str)
            
            # 提取指标
            metrics = {}
            
            # 提取 test_score
            score_matches = re.findall(r"'val/test_score/(\w+)':\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", dict_str)
            for dataset, value in score_matches:
                if 'score' not in metrics:
                    metrics['score'] = {}
                metrics['score'][dataset] = float(value)
            
            # 提取 test_hallucination
            hall_matches = re.findall(r"'val/test_hallucination/(\w+)':\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", dict_str)
            for dataset, value in hall_matches:
                if 'hallucination' not in metrics:
                    metrics['hallucination'] = {}
                metrics['hallucination'][dataset] = float(value)
            
            # 提取 test_n_correct
            correct_matches = re.findall(r"'val/test_n_correct/(\w+)':\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", dict_str)
            for dataset, value in correct_matches:
                if 'correct' not in metrics:
                    metrics['correct'] = {}
                metrics['correct'][dataset] = float(value)
            
            # 提取 test_n_miss
            miss_matches = re.findall(r"'val/test_n_miss/(\w+)':\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", dict_str)
            for dataset, value in miss_matches:
                if 'miss' not in metrics:
                    metrics['miss'] = {}
                metrics['miss'][dataset] = float(value)
            
            # 提取 test_answer_score
            answer_score_matches = re.findall(r"'val/test_answer_score/(\w+)':\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", dict_str)
            for dataset, value in answer_score_matches:
                if 'answer_score' not in metrics:
                    metrics['answer_score'] = {}
                metrics['answer_score'][dataset] = float(value)
            
            # 存储数据
            if metrics:
                for metric_type, datasets in metrics.items():
                    for dataset, value in datasets.items():
                        data[metric_type][dataset].append((current_step, value))
                
            i = j - 1
        
        i += 1
    
    return data


def find_best_step(data):
    """找出三个数据集平均 test_score 最高的 step"""
    if 'score' not in data:
        return None, None, {}
    
    # 计算每个 step 的平均 score
    step_avg_scores = {}
    step_detailed_scores = {}
    
    # 获取所有 step
    all_steps = set()
    for dataset, values in data['score'].items():
        for step, _ in values:
            all_steps.add(step)
    
    # 对每个 step 计算平均分
    for step in sorted(all_steps):
        scores = []
        details = {}
        for dataset, values in data['score'].items():
            for s, v in values:
                if s == step:
                    scores.append(v)
                    details[dataset] = v
                    break
        
        if scores:
            step_avg_scores[step] = sum(scores) / len(scores)
            step_detailed_scores[step] = details
    
    # 找到最高分的 step
    if step_avg_scores:
        best_step = max(step_avg_scores.keys(), key=lambda s: step_avg_scores[s])
        best_avg_score = step_avg_scores[best_step]
        best_details = step_detailed_scores[best_step]
        return best_step, best_avg_score, best_details
    
    return None, None, {}


def plot_combined_metrics(data, output_path):
    """绘制综合指标折线图（所有指标在一起，每个指标一个子图）"""
    
    metric_names = {
        'score': 'Score',
        'hallucination': 'Hallucination',
        'correct': 'Correct',
        'miss': 'Miss'
    }
    
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Metrics Over Steps', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['score', 'hallucination', 'correct', 'miss']
    
    for idx, metric_type in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        if metric_type in data:
            for dataset, values in data[metric_type].items():
                if values:
                    steps = [v[0] for v in values]
                    vals = [v[1] for v in values]
                    ax.plot(steps, vals, marker='o', label=dataset, linewidth=2, markersize=5)
        
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel(metric_names[metric_type], fontsize=11)
        ax.set_title(metric_names[metric_type], fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined plot: {output_path}")
    plt.close()


if __name__ == '__main__':
    # 从命令行参数获取 log 文件路径，或使用默认路径
    log_file = sys.argv[1]
    
    # 获取 log 文件的基础名称（不带扩展名）
    log_basename = os.path.basename(log_file)
    log_name = os.path.splitext(log_basename)[0]
    
    # 确定输出目录和文件名
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # 获取项目根目录（上一级目录）
    output_dir = os.path.join(project_root, 'output_pngs')
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'{log_name}.png')
    
    print(f"Analyzing log file: {log_file}")
    print(f"Output will be saved to: {output_path}\n")
    
    print("Parsing log file...")
    data = parse_log_file(log_file)
    
    print(f"\nFound {len(data)} metric types:")
    for metric_type, datasets in data.items():
        print(f"  {metric_type}: {list(datasets.keys())}")
        for dataset, values in datasets.items():
            print(f"    {dataset}: {len(values)} data points")
    
    # 找出表现最好的 step
    best_step, best_avg_score, best_details = find_best_step(data)
    if best_step is not None:
        print("\n" + "="*70)
        print("🏆 Best Validation Score")
        print("="*70)
        print(f"Step: {best_step}")
        print(f"Average test_score: {best_avg_score:.6f}")
        print("\nDetailed scores by dataset:")
        for dataset, score in sorted(best_details.items()):
            print(f"  {dataset:20s}: {score:.6f}")
        print("="*70)
    
    print("\nGenerating plot...")
    plot_combined_metrics(data, output_path)
    
    print("\n✓ Analysis complete!")

# python scripts/analyze_grpo_log.py A2.log