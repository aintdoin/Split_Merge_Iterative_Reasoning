#!/usr/bin/env python3
"""
Compare reward distributions between think-on and think-off modes
Analyzes JSONL files from inference results and plots bar charts
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os


def load_jsonl(file_path):
    """Load JSONL file and return list of records"""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def analyze_rewards(records):
    """
    Analyze reward distribution by answerable status
    Returns dict: {answerable: {reward: count}}
    """
    stats = {
        True: defaultdict(int),
        False: defaultdict(int)
    }
    
    for record in records:
        answerable = record.get('answerable', True)
        reward = record.get('reward', 0)
        stats[answerable][reward] += 1
    
    return stats


def calculate_percentages(stats):
    """Convert counts to percentages"""
    percentages = {}
    for answerable, reward_counts in stats.items():
        total = sum(reward_counts.values())
        if total > 0:
            percentages[answerable] = {
                reward: (count / total) * 100 
                for reward, count in reward_counts.items()
            }
        else:
            percentages[answerable] = {}
    return percentages


def plot_comparison(think_on_stats, think_off_stats, output_file, dataset_name):
    """
    Plot side-by-side comparison of reward distributions
    """
    # Convert to percentages
    think_on_pct = calculate_percentages(think_on_stats)
    think_off_pct = calculate_percentages(think_off_stats)
    
    # Reward values
    rewards = [-1, 0, 1]
    
    # Create figure with 2 subplots (one for answerable=True, one for False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar width and positions
    bar_width = 0.35
    x = np.arange(len(rewards))
    
    # Colors for think-on and think-off
    color_on = '#4CAF50'  # Green
    color_off = '#2196F3'  # Blue
    
    # Plot for answerable=True
    ax1 = axes[0]
    on_values_true = [think_on_pct.get(True, {}).get(r, 0) for r in rewards]
    off_values_true = [think_off_pct.get(True, {}).get(r, 0) for r in rewards]
    
    bars1 = ax1.bar(x - bar_width/2, on_values_true, bar_width, 
                    label='Think ON', color=color_on, alpha=0.8)
    bars2 = ax1.bar(x + bar_width/2, off_values_true, bar_width,
                    label='Think OFF', color=color_off, alpha=0.8)
    
    ax1.set_xlabel('Reward', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Answerable = True\n({dataset_name})', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(rewards)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 100)
    
    # Add percentage labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9)
    
    # Add sample counts
    on_total_true = sum(think_on_stats.get(True, {}).values())
    off_total_true = sum(think_off_stats.get(True, {}).values())
    ax1.text(0.5, 0.95, f'Think ON: n={on_total_true}, Think OFF: n={off_total_true}',
            transform=ax1.transAxes, ha='center', va='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Plot for answerable=False
    ax2 = axes[1]
    on_values_false = [think_on_pct.get(False, {}).get(r, 0) for r in rewards]
    off_values_false = [think_off_pct.get(False, {}).get(r, 0) for r in rewards]
    
    bars3 = ax2.bar(x - bar_width/2, on_values_false, bar_width,
                    label='Think ON', color=color_on, alpha=0.8)
    bars4 = ax2.bar(x + bar_width/2, off_values_false, bar_width,
                    label='Think OFF', color=color_off, alpha=0.8)
    
    ax2.set_xlabel('Reward', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Answerable = False\n({dataset_name})', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(rewards)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 100)
    
    # Add percentage labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9)
    
    # Add sample counts
    on_total_false = sum(think_on_stats.get(False, {}).values())
    off_total_false = sum(think_off_stats.get(False, {}).values())
    ax2.text(0.5, 0.95, f'Think ON: n={on_total_false}, Think OFF: n={off_total_false}',
            transform=ax2.transAxes, ha='center', va='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def print_statistics(think_on_stats, think_off_stats, dataset_name):
    """Print detailed statistics"""
    print(f"\n{'='*80}")
    print(f"STATISTICS FOR {dataset_name.upper()}")
    print(f"{'='*80}\n")
    
    think_on_pct = calculate_percentages(think_on_stats)
    think_off_pct = calculate_percentages(think_off_stats)
    
    for answerable in [True, False]:
        print(f"\n{'─'*80}")
        print(f"Answerable = {answerable}")
        print(f"{'─'*80}")
        
        # Think ON
        on_total = sum(think_on_stats.get(answerable, {}).values())
        print(f"\nThink ON (n={on_total}):")
        print(f"  Reward  -1: {think_on_stats.get(answerable, {}).get(-1, 0):5d} samples ({think_on_pct.get(answerable, {}).get(-1, 0):5.1f}%)")
        print(f"  Reward   0: {think_on_stats.get(answerable, {}).get(0, 0):5d} samples ({think_on_pct.get(answerable, {}).get(0, 0):5.1f}%)")
        print(f"  Reward  +1: {think_on_stats.get(answerable, {}).get(1, 0):5d} samples ({think_on_pct.get(answerable, {}).get(1, 0):5.1f}%)")
        
        # Think OFF
        off_total = sum(think_off_stats.get(answerable, {}).values())
        print(f"\nThink OFF (n={off_total}):")
        print(f"  Reward  -1: {think_off_stats.get(answerable, {}).get(-1, 0):5d} samples ({think_off_pct.get(answerable, {}).get(-1, 0):5.1f}%)")
        print(f"  Reward   0: {think_off_stats.get(answerable, {}).get(0, 0):5d} samples ({think_off_pct.get(answerable, {}).get(0, 0):5.1f}%)")
        print(f"  Reward  +1: {think_off_stats.get(answerable, {}).get(1, 0):5d} samples ({think_off_pct.get(answerable, {}).get(1, 0):5.1f}%)")
        
        # Calculate difference
        print(f"\nDifference (Think OFF - Think ON):")
        print(f"  Reward  -1: {think_off_pct.get(answerable, {}).get(-1, 0) - think_on_pct.get(answerable, {}).get(-1, 0):+5.1f}%")
        print(f"  Reward   0: {think_off_pct.get(answerable, {}).get(0, 0) - think_on_pct.get(answerable, {}).get(0, 0):+5.1f}%")
        print(f"  Reward  +1: {think_off_pct.get(answerable, {}).get(1, 0) - think_on_pct.get(answerable, {}).get(1, 0):+5.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Compare reward distributions between think-on and think-off modes"
    )
    parser.add_argument(
        "--think-on",
        type=str,
        required=True,
        help="Path to think-on JSONL results file"
    )
    parser.add_argument(
        "--think-off",
        type=str,
        required=True,
        help="Path to think-off JSONL results file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the plot (e.g., 'comparison.png')"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Dataset",
        help="Name of the dataset for plot title"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading think-on data from {args.think_on}...")
    think_on_data = load_jsonl(args.think_on)
    print(f"  Loaded {len(think_on_data)} samples")
    
    print(f"Loading think-off data from {args.think_off}...")
    think_off_data = load_jsonl(args.think_off)
    print(f"  Loaded {len(think_off_data)} samples")
    
    # Analyze rewards
    print("\nAnalyzing reward distributions...")
    think_on_stats = analyze_rewards(think_on_data)
    think_off_stats = analyze_rewards(think_off_data)
    
    # Print statistics
    print_statistics(think_on_stats, think_off_stats, args.dataset_name)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot comparison
    print(f"\nGenerating comparison plot...")
    plot_comparison(think_on_stats, think_off_stats, args.output, args.dataset_name)
    
    print("\nDone!")


if __name__ == '__main__':
    main()