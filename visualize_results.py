#!/usr/bin/env python3
"""
可视化深度上下文赌博机实验结果的工具。

该脚本读取实验结果并生成各种可视化图表,包括:
- 累积遗憾曲线
- 平均奖励对比
- 算法性能热力图
- 逐步奖励和遗憾曲线
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
import glob
from datetime import datetime

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_cumulative_regret_comparison(df: pd.DataFrame, output_dir: str):
    """绘制累积遗憾对比图"""
    
    plt.figure(figsize=(12, 6))
    
    # 按数据集分组
    datasets = df['data_type'].unique()
    
    for dataset in datasets:
        dataset_df = df[df['data_type'] == dataset]
        algorithms = dataset_df['algorithm'].values
        regrets = dataset_df['cumulative_regret'].values
        
        plt.plot(algorithms, regrets, marker='o', label=dataset, linewidth=2)
    
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Cumulative Regret', fontsize=12)
    plt.title('Cumulative Regret Comparison Across Datasets', fontsize=14, fontweight='bold')
    plt.legend(title='Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'cumulative_regret_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"保存图表: {output_file}")
    plt.close()


def plot_average_reward_heatmap(df: pd.DataFrame, output_dir: str):
    """绘制平均奖励热力图"""
    
    pivot_table = df.pivot(index='data_type', columns='algorithm', values='avg_reward')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlGnBu', cbar_kws={'label': 'Average Reward'})
    plt.title('Average Reward Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Dataset', fontsize=12)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'average_reward_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"保存图表: {output_file}")
    plt.close()


def plot_regret_heatmap(df: pd.DataFrame, output_dir: str):
    """绘制累积遗憾热力图"""
    
    pivot_table = df.pivot(index='data_type', columns='algorithm', values='cumulative_regret')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn_r', cbar_kws={'label': 'Cumulative Regret'})
    plt.title('Cumulative Regret Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Dataset', fontsize=12)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'cumulative_regret_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"保存图表: {output_file}")
    plt.close()


def plot_algorithm_ranking(df: pd.DataFrame, output_dir: str):
    """绘制算法排名条形图"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 按累积遗憾排名(越低越好)
    regret_ranking = df.groupby('algorithm')['cumulative_regret'].mean().sort_values()
    axes[0].barh(regret_ranking.index, regret_ranking.values, color='coral')
    axes[0].set_xlabel('Average Cumulative Regret', fontsize=12)
    axes[0].set_title('Algorithm Ranking by Cumulative Regret (Lower is Better)', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    
    # 按平均奖励排名(越高越好)
    reward_ranking = df.groupby('algorithm')['avg_reward'].mean().sort_values(ascending=False)
    axes[1].barh(reward_ranking.index, reward_ranking.values, color='skyblue')
    axes[1].set_xlabel('Average Reward', fontsize=12)
    axes[1].set_title('Algorithm Ranking by Average Reward (Higher is Better)', fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'algorithm_ranking.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"保存图表: {output_file}")
    plt.close()


def plot_step_by_step_regret(detailed_dir: str, output_dir: str, dataset: Optional[str] = None,
                             algorithms: Optional[List[str]] = None, 
                             seed: Optional[int] = None,
                             use_latest: bool = True):
    """绘制逐步累积遗憾曲线"""
    
    if not os.path.exists(detailed_dir):
        print(f"未找到详细日志目录: {detailed_dir}")
        return
    
    # 查找遗憾数据文件
    if seed is not None:
        # 如果指定了seed，查找特定seed的文件
        pattern = f"{dataset}_regrets_seed{seed}_*.csv" if dataset else f"*_regrets_seed{seed}_*.csv"
    else:
        pattern = f"{dataset}_regrets_*.csv" if dataset else "*_regrets_*.csv"
    
    regret_files = glob.glob(os.path.join(detailed_dir, pattern))
    
    if not regret_files:
        print(f"未找到遗憾数据文件: {pattern}")
        return
    
    # 如果use_latest=True且有多个文件，只使用最新的
    if use_latest and len(regret_files) > 1:
        # 按文件修改时间排序，取最新的
        regret_files = [max(regret_files, key=os.path.getctime)]
        print(f"找到 {len(glob.glob(os.path.join(detailed_dir, pattern)))} 个文件，使用最新的")
    
    for regret_file in regret_files:
        df = pd.read_csv(regret_file)
        dataset_name = os.path.basename(regret_file).split('_regrets_')[0]
        
        # 提取seed和时间戳信息（用于文件名）
        file_basename = os.path.basename(regret_file)
        # 例如: linear_regrets_seed42_20260129_143025.csv
        seed_timestamp = file_basename.replace(f'{dataset_name}_regrets_', '').replace('.csv', '')
        
        plt.figure(figsize=(12, 6))
        
        # 获取所有算法列（排除optimal）
        algo_columns = [col for col in df.columns 
                       if col.endswith('_cumulative_regret') and col != 'optimal_cumulative_regret']
        
        # 如果指定了算法列表，只绘制这些算法
        if algorithms:
            algo_columns = [col for col in algo_columns 
                           if col.replace('_cumulative_regret', '') in algorithms]
        
        if not algo_columns:
            print(f"在 {regret_file} 中未找到指定的算法数据")
            continue
        
        # 绘制所有算法的累积遗憾
        for col in algo_columns:
            algo_name = col.replace('_cumulative_regret', '')
            plt.plot(df['step'], df[col], label=algo_name, linewidth=2)
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Cumulative Regret', fontsize=12)
        plt.title(f'{dataset_name} Dataset - Cumulative Regret Curve ({seed_timestamp})', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 输出文件名包含seed和时间戳
        output_file = os.path.join(output_dir, 
                                   f'{dataset_name}_cumulative_regret_curve_{seed_timestamp}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"保存图表: {output_file}")
        plt.close()


def plot_step_by_step_reward(detailed_dir: str, output_dir: str, dataset: Optional[str] = None,
                             algorithms: Optional[List[str]] = None,
                             seed: Optional[int] = None,
                             use_latest: bool = True):
    """绘制逐步累积奖励曲线"""
    
    if not os.path.exists(detailed_dir):
        print(f"未找到详细日志目录: {detailed_dir}")
        return
    
    # 查找奖励数据文件
    if seed is not None:
        pattern = f"{dataset}_rewards_seed{seed}_*.csv" if dataset else f"*_rewards_seed{seed}_*.csv"
    else:
        pattern = f"{dataset}_rewards_*.csv" if dataset else "*_rewards_*.csv"
    
    reward_files = glob.glob(os.path.join(detailed_dir, pattern))
    
    if not reward_files:
        print(f"未找到奖励数据文件: {pattern}")
        return
    
    # 如果use_latest=True且有多个文件，只使用最新的
    if use_latest and len(reward_files) > 1:
        reward_files = [max(reward_files, key=os.path.getctime)]
        print(f"找到 {len(glob.glob(os.path.join(detailed_dir, pattern)))} 个文件，使用最新的")
    
    for reward_file in reward_files:
        df = pd.read_csv(reward_file)
        dataset_name = os.path.basename(reward_file).split('_rewards_')[0]
        
        # 提取seed和时间戳信息
        file_basename = os.path.basename(reward_file)
        seed_timestamp = file_basename.replace(f'{dataset_name}_rewards_', '').replace('.csv', '')
        
        plt.figure(figsize=(12, 6))
        
        # 获取所有算法列（排除optimal）
        algo_columns = [col for col in df.columns 
                       if col.endswith('_cumulative_reward') and col != 'optimal_cumulative_reward']
        
        # 如果指定了算法列表，只绘制这些算法
        if algorithms:
            algo_columns = [col for col in algo_columns 
                           if col.replace('_cumulative_reward', '') in algorithms]
        
        if not algo_columns:
            print(f"在 {reward_file} 中未找到指定的算法数据")
            continue
        
        # 绘制所有算法的累积奖励
        for col in algo_columns:
            algo_name = col.replace('_cumulative_reward', '')
            plt.plot(df['step'], df[col], label=algo_name, linewidth=2)
        
        # 是否包含最优基线
        if 'optimal_cumulative_reward' in df.columns and (not algorithms or 'optimal' in algorithms):
            plt.plot(df['step'], df['optimal_cumulative_reward'], 
                    label='optimal', linestyle='--', color='red', linewidth=2)
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Cumulative Reward', fontsize=12)
        plt.title(f'{dataset_name} Dataset - Cumulative Reward Curve ({seed_timestamp})', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, 
                                   f'{dataset_name}_cumulative_reward_curve_{seed_timestamp}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"保存图表: {output_file}")
        plt.close()


def list_available_detailed_data(results_dir: str):
    """列出所有可用的详细数据文件"""
    detailed_dir = os.path.join(results_dir, 'detailed_logs')
    
    if not os.path.exists(detailed_dir):
        print(f"未找到详细日志目录: {detailed_dir}")
        return
    
    regret_files = glob.glob(os.path.join(detailed_dir, '*_regrets_*.csv'))
    
    if not regret_files:
        print("未找到任何详细数据文件")
        return
    
    print("\n" + "="*60)
    print("可用的详细数据文件:")
    print("="*60)
    
    # 按数据集分组
    data_by_dataset = {}
    for file in regret_files:
        basename = os.path.basename(file)
        dataset_name = basename.split('_regrets_')[0]
        
        # 提取seed和时间戳
        seed_timestamp = basename.replace(f'{dataset_name}_regrets_', '').replace('.csv', '')
        
        if dataset_name not in data_by_dataset:
            data_by_dataset[dataset_name] = []
        
        # 获取文件修改时间
        mod_time = datetime.fromtimestamp(os.path.getctime(file))
        
        data_by_dataset[dataset_name].append({
            'seed_timestamp': seed_timestamp,
            'mod_time': mod_time,
            'file': file
        })
    
    # 打印信息
    for dataset, files in sorted(data_by_dataset.items()):
        print(f"\n📊 数据集: {dataset}")
        # 按时间排序
        files.sort(key=lambda x: x['mod_time'], reverse=True)
        for i, file_info in enumerate(files):
            marker = "⭐ [最新]" if i == 0 else "  "
            print(f"  {marker} {file_info['seed_timestamp']} (修改时间: {file_info['mod_time'].strftime('%Y-%m-%d %H:%M:%S')})")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='可视化深度上下文赌博机实验结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成所有图表（默认使用最新数据）
  python3 visualize_results.py
  
  # 只生成逐步曲线
  python3 visualize_results.py --plot_types step_regret step_reward
  
  # 针对特定数据集和算法
  python3 visualize_results.py --dataset linear --algorithms neural_bandit neural_linear
  
  # 使用特定seed的数据
  python3 visualize_results.py --seed 42 --dataset linear
  
  # 绘制所有历史数据（不只是最新的）
  python3 visualize_results.py --use_all_data
  
  # 列出所有可用的详细数据文件
  python3 visualize_results.py --list_data
        """
    )
    
    # 基本参数
    parser.add_argument('--results_dir', type=str, default='results',
                       help='结果目录路径 (默认: results)')
    parser.add_argument('--output_dir', type=str, default='results/plots',
                       help='图表输出目录 (默认: results/plots)')
    parser.add_argument('--result_file', type=str, default=None,
                       help='指定结果CSV文件 (默认: 使用最新的)')
    
    # 数据筛选参数
    parser.add_argument('--dataset', type=str, default=None,
                       help='指定数据集名称 (例如: linear, mushroom)')
    parser.add_argument('--algorithms', nargs='+', default=None,
                       help='指定要绘制的算法列表 (例如: neural_bandit neural_linear)')
    parser.add_argument('--seed', type=int, default=None,
                       help='指定随机种子，只绘制特定seed的数据')
    parser.add_argument('--use_all_data', action='store_true',
                       help='绘制所有历史数据，而不只是最新的')
    
    # 图表类型选择参数
    parser.add_argument('--plot_types', nargs='+', 
                       choices=['comparison', 'reward_heatmap', 'regret_heatmap', 
                               'ranking', 'step_regret', 'step_reward', 'all'],
                       default=['all'],
                       help="""选择要生成的图表类型:
                       comparison - 累积遗憾对比图
                       reward_heatmap - 平均奖励热力图
                       regret_heatmap - 累积遗憾热力图
                       ranking - 算法排名图
                       step_regret - 逐步累积遗憾曲线
                       step_reward - 逐步累积奖励曲线
                       all - 生成所有图表 (默认)""")
    
    # 输出格式参数
    parser.add_argument('--format', type=str, choices=['png', 'pdf', 'svg'], 
                       default='png',
                       help='输出图表格式 (默认: png)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='图表分辨率 (默认: 300)')
    
    # 工具功能
    parser.add_argument('--list_data', action='store_true',
                       help='列出所有可用的详细数据文件并退出')
    
    args = parser.parse_args()
    
    # 如果是列出数据，执行后退出
    if args.list_data:
        list_available_detailed_data(args.results_dir)
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定要绘制的图表类型
    plot_types = set(args.plot_types)
    if 'all' in plot_types:
        plot_types = {'comparison', 'reward_heatmap', 'regret_heatmap', 
                     'ranking', 'step_regret', 'step_reward'}
    
    print("="*60)
    print("深度上下文赌博机实验结果可视化")
    print("="*60)
    
    # 是否需要汇总数据
    need_summary_data = plot_types & {'comparison', 'reward_heatmap', 'regret_heatmap', 'ranking'}
    
    # 是否需要逐步数据
    need_step_data = plot_types & {'step_regret', 'step_reward'}
    
    df = None
    
    # 读取汇总数据
    if need_summary_data:
        if args.result_file:
            result_file = args.result_file
        else:
            result_files = glob.glob(os.path.join(args.results_dir, 'results_*.csv'))
            if not result_files:
                print(f"\n⚠️  未在 {args.results_dir} 中找到结果文件")
                if need_step_data:
                    print("继续生成逐步数据图表...")
                else:
                    return
            else:
                result_file = max(result_files, key=os.path.getctime)
                print(f"\n📁 读取结果文件: {result_file}")
                df = pd.read_csv(result_file)
                
                # 过滤成功的实验
                df = df[df['status'] == 'success']
                
                if df.empty:
                    print("⚠️  没有成功的实验结果可供可视化")
                    if need_step_data:
                        print("继续生成逐步数据图表...")
                    else:
                        return
                else:
                    print(f"✓ 找到 {len(df)} 条成功的实验结果")
    
    print("\n" + "="*60)
    print("生成可视化图表")
    print("="*60)
    
    # 生成汇总图表
    if df is not None and need_summary_data:
        print("\n📊 生成汇总图表...")
        
        if 'comparison' in plot_types:
            print("  - 累积遗憾对比图")
            plot_cumulative_regret_comparison(df, args.output_dir)
        
        if 'reward_heatmap' in plot_types:
            print("  - 平均奖励热力图")
            plot_average_reward_heatmap(df, args.output_dir)
        
        if 'regret_heatmap' in plot_types:
            print("  - 累积遗憾热力图")
            plot_regret_heatmap(df, args.output_dir)
        
        if 'ranking' in plot_types:
            print("  - 算法排名图")
            plot_algorithm_ranking(df, args.output_dir)
    
    # 生成逐步数据图表
    if need_step_data:
        detailed_dir = os.path.join(args.results_dir, 'detailed_logs')
        if os.path.exists(detailed_dir):
            print("\n📈 生成逐步数据图表...")
            
            if args.dataset:
                print(f"  - 数据集: {args.dataset}")
            if args.algorithms:
                print(f"  - 算法: {', '.join(args.algorithms)}")
            if args.seed is not None:
                print(f"  - 随机种子: {args.seed}")
            if not args.use_all_data:
                print(f"  - 模式: 使用最新数据")
            else:
                print(f"  - 模式: 使用所有历史数据")
            
            use_latest = not args.use_all_data
            
            if 'step_regret' in plot_types:
                print("  - 逐步累积遗憾曲线")
                plot_step_by_step_regret(detailed_dir, args.output_dir, 
                                        args.dataset, args.algorithms,
                                        args.seed, use_latest)
            
            if 'step_reward' in plot_types:
                print("  - 逐步累积奖励曲线")
                plot_step_by_step_reward(detailed_dir, args.output_dir, 
                                        args.dataset, args.algorithms,
                                        args.seed, use_latest)
        else:
            print(f"\n⚠️  未找到详细日志目录: {detailed_dir}")
    
    print("\n" + "="*60)
    print("✅ 可视化完成!")
    print(f"📂 所有图表已保存到: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()