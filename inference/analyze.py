#!/usr/bin/env python3
"""
预实验结果分析程序
分析模型在可回答/不可回答问题上的IDK表现，以及输出长度与IDK率的关系
"""

import json
import argparse
import os
from collections import defaultdict
from typing import List, Dict, Tuple
import statistics

# 尝试导入numpy和matplotlib，如果不存在则使用备选方案
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    # 设置中文字体支持
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib未安装，将跳过图表生成功能")
    print("如需生成图表，请运行: pip install matplotlib")


def load_results(jsonl_path: str) -> List[Dict]:
    """加载JSONL结果文件"""
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def count_tokens(text: str) -> int:
    """
    简单的token计数（按空格分词）
    对于更精确的计数，可以使用tokenizer
    """
    # 简单估算：英文按空格分词，中文按字符数
    words = text.split()
    # 粗略估算：每个词约1.3个token
    return int(len(words) * 1.3)


def analyze_reward_by_answerable(results: List[Dict]) -> Dict:
    """按answerable分类统计各reward值的比例"""
    answerable_true = [r for r in results if r.get('answerable') is True]
    answerable_false = [r for r in results if r.get('answerable') is False]
    
    # answerable=True: 统计reward=1, 0, -1
    reward_true_counts = {1: 0, 0: 0, -1: 0}
    for r in answerable_true:
        reward = r.get('reward', 0)
        if reward in reward_true_counts:
            reward_true_counts[reward] += 1
    
    # answerable=False: 统计reward=1, -1
    reward_false_counts = {1: 0, -1: 0}
    for r in answerable_false:
        reward = r.get('reward', 0)
        if reward in reward_false_counts:
            reward_false_counts[reward] += 1
    
    total_true = len(answerable_true) if answerable_true else 1
    total_false = len(answerable_false) if answerable_false else 1
    
    return {
        'answerable_true': {
            'total': len(answerable_true),
            'reward_1_count': reward_true_counts[1],
            'reward_1_rate': reward_true_counts[1] / total_true,
            'reward_0_count': reward_true_counts[0],
            'reward_0_rate': reward_true_counts[0] / total_true,
            'reward_-1_count': reward_true_counts[-1],
            'reward_-1_rate': reward_true_counts[-1] / total_true,
        },
        'answerable_false': {
            'total': len(answerable_false),
            'reward_1_count': reward_false_counts[1],
            'reward_1_rate': reward_false_counts[1] / total_false,
            'reward_-1_count': reward_false_counts[-1],
            'reward_-1_rate': reward_false_counts[-1] / total_false,
        }
    }





def plot_pie_charts(reward_stats: Dict, output_dir: str, filename_prefix: str):
    """绘制answerable=True和False两种情况下的回答正确率饼图
    
    Args:
        reward_stats: 按answerable分类的reward统计
        output_dir: 输出目录
        filename_prefix: 文件名前缀
    """
    
    if not HAS_MATPLOTLIB:
        print("  跳过饼图生成（matplotlib未安装）")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图: Answerable=True (应该回答的问题)
    ax1 = axes[0]
    stats_true = reward_stats['answerable_true']
    
    if stats_true['total'] > 0:
        labels = ['Correct Answer\n(Reward=1)', 
                  'Said "I don\'t know"\n(Reward=0)', 
                  'Wrong Answer\n(Reward=-1)']
        sizes = [
            stats_true['reward_1_count'],
            stats_true['reward_0_count'],
            stats_true['reward_-1_count']
        ]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # 绿色、橙色、红色
        explode = (0.05, 0.05, 0.05)  # 稍微分离每个扇区
        
        wedges, texts, autotexts = ax1.pie(
            sizes, 
            explode=explode,
            labels=labels, 
            colors=colors,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes))})',
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'},
            shadow=True
        )
        
        # 美化百分比文字
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        
        ax1.set_title(
            f'Answerable=True (Should Answer)\nTotal: {stats_true["total"]} samples',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
    else:
        ax1.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                fontsize=20, transform=ax1.transAxes)
        ax1.set_title('Answerable=True (Should Answer)', fontsize=14, fontweight='bold')
    
    # 右图: Answerable=False (不应该回答的问题)
    ax2 = axes[1]
    stats_false = reward_stats['answerable_false']
    
    if stats_false['total'] > 0:
        labels = ['Correct Refusal\n(Said "I don\'t know")\n(Reward=1)', 
                  'Wrong Answer\n(Should not answer)\n(Reward=-1)']
        sizes = [
            stats_false['reward_1_count'],
            stats_false['reward_-1_count']
        ]
        colors = ['#2ecc71', '#e74c3c']  # 绿色、红色
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = ax2.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes))})',
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'},
            shadow=True
        )
        
        # 美化百分比文字
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        
        ax2.set_title(
            f'Answerable=False (Should Not Answer)\nTotal: {stats_false["total"]} samples',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
    else:
        ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                fontsize=20, transform=ax2.transAxes)
        ax2.set_title('Answerable=False (Should Not Answer)', fontsize=14, fontweight='bold')
    
    plt.suptitle('Model Performance Analysis by Answerability', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{filename_prefix}_pie_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  饼图已保存: {output_path}")
    plt.close()





def print_summary(results: List[Dict], reward_stats: Dict):
    """打印分析摘要"""
    print("\n" + "=" * 80)
    print("预实验结果分析报告 - Reward分布分析")
    print("=" * 80)
    
    # 计算总分（所有样本reward的平均值）
    all_rewards = [r.get('reward', 0) for r in results]
    total_score = sum(all_rewards) / len(all_rewards) if all_rewards else 0
    
    print(f"\n【1. 基本统计】")
    print(f"  总样本数: {len(results)}")
    print(f"  总分 (平均Reward): {total_score:.4f}")
    
    print(f"\n【2. 按Answerable分类的Reward分布】")
    print(f"\n  Answerable=True (应该回答的问题):")
    stats_true = reward_stats['answerable_true']
    print(f"    总样本数:        {stats_true['total']}")
    print(f"    Reward=1:        {stats_true['reward_1_count']:4d} ({stats_true['reward_1_rate']*100:5.2f}%) - 正确回答")
    print(f"    Reward=0:        {stats_true['reward_0_count']:4d} ({stats_true['reward_0_rate']*100:5.2f}%) - 不应拒答但拒答")
    print(f"    Reward=-1:       {stats_true['reward_-1_count']:4d} ({stats_true['reward_-1_rate']*100:5.2f}%) - 错误回答")
    
    print(f"\n  Answerable=False (不应该回答的问题):")
    stats_false = reward_stats['answerable_false']
    print(f"    总样本数:        {stats_false['total']}")
    print(f"    Reward=1:        {stats_false['reward_1_count']:4d} ({stats_false['reward_1_rate']*100:5.2f}%) - 正确拒答")
    print(f"    Reward=-1:       {stats_false['reward_-1_count']:4d} ({stats_false['reward_-1_rate']*100:5.2f}%) - 不应回答但回答")
    
    print(f"\n【3. Token长度统计】")
    all_token_counts = [count_tokens(r.get('model_output', '')) for r in results]
    if HAS_NUMPY:
        mean_tokens = np.mean(all_token_counts)
        median_tokens = np.median(all_token_counts)
        min_tokens = np.min(all_token_counts)
        max_tokens = np.max(all_token_counts)
    else:
        mean_tokens = statistics.mean(all_token_counts) if all_token_counts else 0
        median_tokens = statistics.median(all_token_counts) if all_token_counts else 0
        min_tokens = min(all_token_counts) if all_token_counts else 0
        max_tokens = max(all_token_counts) if all_token_counts else 0
    
    print(f"    平均输出token数: {mean_tokens:.1f}")
    print(f"    中位数:          {median_tokens:.1f}")
    print(f"    最小值:          {min_tokens}")
    print(f"    最大值:          {max_tokens}")
    
    print("\n" + "=" * 80)
    
    # 在报告末尾再次突出显示总分
    all_rewards = [r.get('reward', 0) for r in results]
    total_score = sum(all_rewards) / len(all_rewards) if all_rewards else 0
    
    print(f"\n{'🏆 总分 (平均Reward)':^80s}")
    print(f"{'=' * 80}")
    print(f"{total_score:^80.4f}")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='分析预实验结果：IDK表现分析'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入JSONL文件路径 (例如: prelininary/inference_results/qwen-7b_inference_results.jsonl)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='prelininary/analysis_results',
        help='输出目录 (默认: prelininary/analysis_results)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help='输出文件名前缀 (默认: 从输入文件名提取)'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定输出文件名前缀
    if not args.prefix:
        # 从输入文件名提取（去掉路径和扩展名）
        basename = os.path.basename(args.input)
        args.prefix = os.path.splitext(basename)[0]
    
    print(f"\n正在加载主数据: {args.input}")
    results = load_results(args.input)
    print(f"  已加载 {len(results)} 个样本")
    
    print(f"\n正在分析Reward分布...")
    reward_stats = analyze_reward_by_answerable(results)
    
    print(f"\n正在生成图表...")
    print(f"  生成饼图...")
    plot_pie_charts(reward_stats, args.output_dir, args.prefix)
    
    # 打印摘要
    print_summary(results, reward_stats)
    
    print(f"\n✓ 分析完成！")


if __name__ == '__main__':
    main()