#!/usr/bin/env python3
"""
Batch experiment runner for Deep Contextual Bandits.

This script runs all available algorithms on all datasets and saves the results
to a CSV file for analysis and comparison.

Inspired by the original TensorFlow implementation:
https://github.com/tensorflow/models/tree/archive/research/deep_contextual_bandits
"""

import argparse
import time
import numpy as np
import os
import csv
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

# Import our PyTorch implementations
from bandits.algorithms.neural_bandit_model import NeuralBanditModel
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.algorithms.uniform_sampling import UniformSampling
from bandits.algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from bandits.algorithms.bootstrapped_bnn_sampling import BootstrappedBNNSampling
from bandits.algorithms.fixed_policy_sampling import FixedPolicySampling
from bandits.algorithms.ucb1_sampling import UCB1Sampling
from bandits.algorithms.linucb_sampling import LinUCBSampling
from bandits.algorithms.epsilon_greedy_sampling import EpsilonGreedySampling
from bandits.algorithms.neural_ucb_sampling import NeuralUCBSampling
from bandits.algorithms.neural_linucb_sampling import NeuralLinUCBSampling

# Import core functionality
from bandits.core.contextual_bandit import run_contextual_bandit

# Import data samplers
from bandits.data.synthetic_data_sampler import (
    sample_linear_data, 
    sample_sparse_linear_data, 
    sample_wheel_bandit_data
)
from bandits.data.data_sampler import (
    sample_mushroom_data, sample_stock_data, sample_jester_data,
    sample_statlog_data, sample_adult_data, sample_census_data,
    sample_census_data, sample_statlog_shuttle_data, sample_newsgroups_data
)

# ================ 数据采样与算法创建 ==================
# 根据数据类型采样数据集，可以控制数据采样的参数
# 通过ynthetic_data_sampler.py和data_sampler.py中的函数进行采样
def sample_data(data_type: str, num_contexts: int, data_dir: str = "datasets") -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Sample data from given 'data_type'."""
    
    if data_type == 'linear':
        num_actions = 8
        context_dim = 10
        noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
        dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim, num_actions, sigma=noise_stds)
        opt_rewards, opt_actions = opt_linear
        
    elif data_type == 'sparse_linear':
        num_actions = 7
        context_dim = 10
        noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
        num_nnz_dims = int(context_dim / 3.0)
        dataset, _, opt_sparse_linear = sample_sparse_linear_data(
            num_contexts, context_dim, num_actions, num_nnz_dims, sigma=noise_stds)
        opt_rewards, opt_actions = opt_sparse_linear
        
    elif data_type == 'wheel':
        delta = 0.95
        num_actions = 5
        context_dim = 2
        mean_v = [1.0, 1.0, 1.0, 1.0, 1.2]
        std_v = [0.05, 0.05, 0.05, 0.05, 0.05]
        mu_large = 50
        std_large = 0.01
        dataset, opt_wheel = sample_wheel_bandit_data(num_contexts, delta, mean_v, std_v, mu_large, std_large)
        opt_rewards, opt_actions = opt_wheel
        
    elif data_type == 'mushroom':
        num_actions = 2
        num_contexts = min(8124, num_contexts)
        sampled_vals = sample_mushroom_data(num_contexts, r_noeat=0, r_eat_safe=5, r_eat_poison_bad=-35, r_eat_poison_good=5, prob_poison_bad=0.5)
        dataset, opt_vals = sampled_vals
        opt_rewards, opt_actions = opt_vals
        context_dim = dataset.shape[1] - num_actions
        
    elif data_type == 'financial':
        num_actions = 8
        context_dim = 21
        num_contexts = min(3713, num_contexts)
        file_name = os.path.join(data_dir, 'raw_stock_contexts')
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Financial dataset not found at {file_name}.")
        dataset, opt_vals = sample_stock_data(file_name, context_dim, num_actions, num_contexts, sigma=0.01, shuffle_rows=True)
        opt_rewards, opt_actions = opt_vals
        
    elif data_type == 'jester':
        num_actions = 8
        context_dim = 32
        num_contexts = min(19181, num_contexts)
        file_name = os.path.join(data_dir, 'jester_data_40jokes_19181users.npy')
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Jester dataset not found at {file_name}.")
        dataset, opt_jester = sample_jester_data(file_name, context_dim, num_actions, num_contexts, shuffle_rows=True, shuffle_cols=True)
        opt_rewards, opt_actions = opt_jester
        
    elif data_type == 'statlog':
        file_name = os.path.join(data_dir, 'statlog.trn')
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Statlog dataset not found at {file_name}.")
        num_actions = 7
        num_contexts = min(43500, num_contexts)
        dataset, (opt_rewards, opt_actions) = sample_statlog_data(file_name, num_contexts, shuffle_rows=True)
        context_dim = dataset.shape[1] - num_actions #dataset为横向拼接上下文和奖励矩阵后的结果，因此需要减去动作数量才能得到上下文维度
        
    elif data_type == 'adult':
        num_contexts = min(48842, num_contexts)
        dataset, (opt_rewards, opt_actions) = sample_adult_data(num_contexts, shuffle_rows=True)
        num_actions = len(np.unique(opt_actions))
        context_dim = dataset.shape[1] - num_actions
        
    elif data_type == 'covertype':
        num_contexts = min(581012, num_contexts)
        dataset, (opt_rewards, opt_actions) = sample_covertype_data(num_contexts, shuffle_rows=True)
        num_actions = len(np.unique(opt_actions))
        context_dim = dataset.shape[1] - num_actions
        
    elif data_type == 'census':
        num_contexts = min(2458285, num_contexts)
        dataset, (opt_rewards, opt_actions) = sample_census_data(num_contexts, shuffle_rows=True)
        num_actions = len(np.unique(opt_actions))
        context_dim = dataset.shape[1] - num_actions
        
    elif data_type == 'statlog_shuttle':
        num_contexts = min(58000, num_contexts)
        dataset, (opt_rewards, opt_actions) = sample_statlog_shuttle_data(num_contexts, shuffle_rows=True)
        num_actions = len(np.unique(opt_actions))
        context_dim = dataset.shape[1] - num_actions

    elif data_type == 'newsgroups':
        file_name = os.path.join(data_dir, 'newsgroups.npz')
        if not os.path.exists(file_name):
            raise FileNotFoundError(
                f"Newsgroups dataset not found at {file_name}. "
                "Run prepare_newsgroups.py first.")
        num_contexts = min(5851, num_contexts)
        dataset, (opt_rewards, opt_actions) = sample_newsgroups_data(
            file_name, num_contexts, shuffle_rows=True)
        num_actions = 6
        context_dim = dataset.shape[1] - num_actions
        
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    return dataset, opt_rewards, opt_actions, num_actions, context_dim # 返回数据集（横向拼接上下文和奖励）、最优奖励、最优动作、动作数量、上下文维度

# 根据算法名称创建算法实例，传入算法名称、动作数量和上下文维度
def create_algorithm(name: str, num_actions: int, context_dim: int) -> Any:
    """Create algorithm instance based on name."""
    
    if name == 'neural_bandit':
        hparams = {
            "context_dim": context_dim,
            "num_actions": num_actions,
            "layer_sizes": [100, 100],
            "activation": "relu",
            "initial_lr": 0.001,
            "batch_size": 512,
            "init_scale": 0.3,
            "use_dropout": False,
            "dropout_rate": 0.1,
            "layer_norm": False,
            "verbose": False
        }
        return NeuralBanditModel(hparams, name="neural_bandit")
        
    elif name == 'neural_linear':
        hparams = {
            "context_dim": context_dim, #上下文维度
            "num_actions": num_actions, #动作数量
            "layer_sizes": [100, 100], #神经网络隐藏层层数、大小
            "activation": "relu", #激活函数
            "initial_lr": 0.001, #初始学习率
            "batch_size": 512, #批量大小
            "init_scale": 0.3, #权重初始化尺度
            "use_dropout": False, #是否使用dropout
            "dropout_rate": 0.1, #dropout比率
            "layer_norm": False, #是否使用层归一化
            "verbose": False, #是否打印详细日志
            "lambda_prior": 0.25, #线性层先验正则化参数
            "a0": 6, #线性层先验参数
            "b0": 6, #线性层先验参数
            "training_freq": 100, #神经网络训练频率
            "training_freq_network": 100, #神经网络训练频率
            "training_epochs": 100, #神经网络训练轮数
            "initial_pulls": 2 #初始拉动次数
        }
        return NeuralLinearPosteriorSampling(hparams, name="neural_linear")
        
    elif name == 'uniform':
        # Create a simple hparams object for uniform sampling
        class HParams:
            def __init__(self, num_actions):
                self.num_actions = num_actions
        
        hparams = HParams(num_actions)
        return UniformSampling("uniform", hparams)
        
    elif name == 'linear_full_posterior':
        # Create a simple hparams object for linear full posterior
        class HParams:
            def __init__(self, context_dim, num_actions):
                self.context_dim = context_dim
                self.num_actions = num_actions
                self.lambda_prior = 0.25
                self.a0 = 6
                self.b0 = 6
                self.initial_pulls = 2
        
        hparams = HParams(context_dim, num_actions)
        return LinearFullPosteriorSampling("linear_full_posterior", hparams)
        
    elif name == 'bootstrapped_bnn':
        # Create a simple hparams object for bootstrapped BNN
        class HParams:
            def __init__(self, context_dim, num_actions):
                self.context_dim = context_dim
                self.num_actions = num_actions
                self.training_freq = 100
                self.training_epochs = 100
                self.q = 10  # number of models
                self.p = 0.8  # probability of including each datapoint
                self.initial_pulls = 2
                self.buffer_s = 10000
        
        hparams = HParams(context_dim, num_actions)
        return BootstrappedBNNSampling("bootstrapped_bnn", hparams)
        
    elif name == 'fixed_policy':
        # Create a simple hparams object for fixed policy
        class HParams:
            def __init__(self, num_actions):
                self.num_actions = num_actions
        
        hparams = HParams(num_actions)
        # Create uniform policy
        p = np.ones(num_actions) / num_actions
        return FixedPolicySampling("fixed_policy", p, hparams)

    elif name == 'ucb1':
        class HParams:
            def __init__(self, num_actions):
                self.num_actions = num_actions
                self.alpha = 1.0
        hparams = HParams(num_actions)
        return UCB1Sampling("ucb1", hparams)

    elif name == 'linucb':
        class HParams:
            def __init__(self, context_dim, num_actions):
                self.context_dim = context_dim
                self.num_actions = num_actions
                self.alpha = 1.0
                self.lambda_prior = 1.0
        hparams = HParams(context_dim, num_actions)
        return LinUCBSampling("linucb", hparams)

    elif name == 'epsilon_greedy':
        class HParams:
            def __init__(self, num_actions):
                self.num_actions = num_actions
                self.epsilon = 0.1
                self.epsilon_decay = 0.0
        hparams = HParams(num_actions)
        return EpsilonGreedySampling("epsilon_greedy", hparams)

    elif name == 'neural_ucb':
        hparams = {
            "context_dim": context_dim,
            "num_actions": num_actions,
            "layer_sizes": [100, 100],
            "activation": "relu",
            "initial_lr": 0.001,
            "batch_size": 512,
            "init_scale": 0.3,
            "use_dropout": False,
            "dropout_rate": 0.1,
            "layer_norm": False,
            "verbose": False,
            "alpha": 1.0,
            "lambda_prior": 1.0,
            "training_freq": 100,
            "training_freq_network": 100,
            "training_epochs": 100,
            "initial_pulls": 2
        }
        return NeuralUCBSampling(hparams, name="neural_ucb")

    elif name == 'neural_linucb':
        hparams = {
            "context_dim": context_dim,
            "num_actions": num_actions,
            "layer_sizes": [100, 100],
            "activation": "relu",
            "initial_lr": 0.001,
            "batch_size": 512,
            "init_scale": 0.3,
            "use_dropout": False,
            "dropout_rate": 0.1,
            "layer_norm": False,
            "verbose": False,
            "alpha": 1.0,
            "lambda_prior": 0.25,
            "training_freq": 100,
            "training_freq_network": 100,
            "training_epochs": 100,
            "initial_pulls": 2
        }
        return NeuralLinUCBSampling(hparams, name="neural_linucb")

    else:
        raise ValueError(f"Unknown algorithm: {name}")


def evaluate_performance(actions: np.ndarray, rewards: np.ndarray, opt_rewards: np.ndarray, opt_actions: np.ndarray) -> Dict[str, float]:
    """Evaluate algorithm performance."""
    num_algorithms = actions.shape[1]
    metrics = {}
    
    for i in range(num_algorithms):
        algo_actions = actions[:, i]
        algo_rewards = rewards[:, i]
        
        # Cumulative regret
        cumulative_regret = np.sum(opt_rewards - algo_rewards)
        
        # Average reward
        avg_reward = np.mean(algo_rewards)
        
        # Regret at each step
        regret = opt_rewards - algo_rewards
        cumulative_regret_steps = np.cumsum(regret)
        
        # Store metrics
        metrics[f'algo_{i}_cumulative_regret'] = cumulative_regret
        metrics[f'algo_{i}_avg_reward'] = avg_reward
        metrics[f'algo_{i}_final_regret'] = cumulative_regret_steps[-1]
    
    return metrics


def save_combined_step_data(
    data_type: str,
    algorithms_data: Dict[str, Dict],
    output_dir: str,
    seed: int,
    num_contexts: int = 0
):
    """
    将同一数据集的所有算法数据保存到一个文件。
    
    Args:
        data_type: 数据集名称
        algorithms_data: {algorithm_name: {'rewards': array, 'actions': array, ...}}
        output_dir: 输出目录
        seed: 随机种子
        num_contexts: 运行轮数
    """
    
    detailed_dir = os.path.join(output_dir, 'detailed_logs')
    os.makedirs(detailed_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建奖励数据文件
    reward_filename = f"{timestamp}_{num_contexts}r_{data_type}_rewards_seed{seed}.csv"
    reward_filepath = os.path.join(detailed_dir, reward_filename)
    
    # 创建遗憾数据文件
    regret_filename = f"{timestamp}_{num_contexts}r_{data_type}_regrets_seed{seed}.csv"
    regret_filepath = os.path.join(detailed_dir, regret_filename)
    
    # 构建奖励DataFrame
    reward_data = {'step': np.arange(len(list(algorithms_data.values())[0]['rewards']))}
    regret_data = {'step': reward_data['step'].copy()}
    
    for algo_name, data in algorithms_data.items():
        reward_data[f'{algo_name}_reward'] = data['rewards']
        reward_data[f'{algo_name}_cumulative_reward'] = np.cumsum(data['rewards'])
        
        step_regret = data['opt_rewards'] - data['rewards']
        regret_data[f'{algo_name}_step_regret'] = step_regret
        regret_data[f'{algo_name}_cumulative_regret'] = np.cumsum(step_regret)
    
    # 添加最优基线
    first_algo = list(algorithms_data.values())[0]
    reward_data['optimal_reward'] = first_algo['opt_rewards']
    reward_data['optimal_cumulative_reward'] = np.cumsum(first_algo['opt_rewards'])
    
    regret_data['optimal_reward'] = first_algo['opt_rewards']
    
    # 保存文件
    pd.DataFrame(reward_data).to_csv(reward_filepath, index=False)
    pd.DataFrame(regret_data).to_csv(regret_filepath, index=False)
    
    print(f"    Saved combined rewards to: {reward_filepath}")
    print(f"    Saved combined regrets to: {regret_filepath}")


def save_step_by_step_data(
    data_type: str,
    algorithm_name: str,
    actions: np.ndarray,
    rewards: np.ndarray,
    opt_rewards: np.ndarray,
    opt_actions: np.ndarray,
    output_dir: str,
    seed: int,
    num_contexts: int = 0
):
    """
    保存每一步的详细数据。
    
    Args:
        data_type: 数据集名称
        algorithm_name: 算法名称
        actions: 算法选择的动作序列 (num_contexts,)
        rewards: 算法获得的奖励序列 (num_contexts,)
        opt_rewards: 最优奖励序列 (num_contexts,)
        opt_actions: 最优动作序列 (num_contexts,)
        output_dir: 输出目录
        seed: 随机种子
        num_contexts: 运行轮数
    """
    
    # 创建详细数据子目录
    detailed_dir = os.path.join(output_dir, 'detailed_logs')
    os.makedirs(detailed_dir, exist_ok=True)
    
    # 生成文件名: timestamp_轮数r_dataset_algorithm_seed.csv
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{num_contexts}r_{data_type}_{algorithm_name}_seed{seed}.csv"
    filepath = os.path.join(detailed_dir, filename)
    
    # 计算每一步的指标
    step_regret = opt_rewards - rewards  # 每步遗憾
    cumulative_regret = np.cumsum(step_regret)  # 累积遗憾
    cumulative_reward = np.cumsum(rewards)  # 累积奖励
    is_optimal = (actions == opt_actions).astype(int)  # 是否选择最优动作
    
    # 创建DataFrame
    df = pd.DataFrame({
        'step': np.arange(len(actions)),
        'action': actions,
        'reward': rewards,
        'opt_action': opt_actions,
        'opt_reward': opt_rewards,
        'is_optimal': is_optimal,
        'step_regret': step_regret,
        'cumulative_regret': cumulative_regret,
        'cumulative_reward': cumulative_reward
    })
    
    # 保存CSV
    df.to_csv(filepath, index=False)
    print(f"    Saved step-by-step data to: {filepath}")

# ================= 主运行逻辑 ==================
# 传入参数：数据类型、算法名称、上下文数量/轮数、数据目录、随机种子、输出目录
def run_single_experiment(data_type: str, algorithm_name: str, num_contexts: int, 
                         data_dir: str, seed: int, output_dir: str) -> Dict[str, Any]:
    """Run a single experiment and return results."""
    
    print(f"Running {algorithm_name} on {data_type} dataset...")
    
    # Set random seed
    # 固定随机种子
    np.random.seed(seed)
    
    try:
        # Sample data
        # 采样数据集，输入数据类型、上下文数量/轮数，数据目录
        # 返回数据集（）、最优奖励、最优动作、动作数量、上下文维度
        start_time = time.time()
        dataset, opt_rewards, opt_actions, num_actions, context_dim = sample_data(
            data_type, num_contexts, data_dir)
        data_time = time.time() - start_time # 采样数据时间
        
        # Create algorithm
        # 根据算法名称、动作数量、上下文维度，创建不同算法实例
        algorithm = create_algorithm(algorithm_name, num_actions, context_dim)
        
        # Run experiment
        start_time = time.time()
        # 运行上下文bandit实验，返回动作序列（num_contexts，算法数量）和奖励序列（num_contexts，算法数量），传入上下文维度、动作数量、数据集、算法实例
        actions, rewards = run_contextual_bandit(context_dim, num_actions, dataset, [algorithm]) 
        experiment_time = time.time() - start_time # 实验运行时间

        # ========== 新增：保存逐步数据 ==========
        save_step_by_step_data(
            data_type=data_type,
            algorithm_name=algorithm_name,
            actions=actions[:, 0],  # 取第一个算法的动作序列
            rewards=rewards[:, 0],  # 取第一个算法的奖励序列
            opt_rewards=opt_rewards,
            opt_actions=opt_actions,
            output_dir=output_dir,
            seed=seed,
            num_contexts=num_contexts
        )

         # 准备返回的逐步数据
        step_data = {
            'rewards': rewards[:, 0],
            'actions': actions[:, 0],
            'opt_rewards': opt_rewards,
            'opt_actions': opt_actions
        }

        # ========================================
        
        # Evaluate performance
        metrics = evaluate_performance(actions, rewards, opt_rewards, opt_actions)
        
        # Prepare results
        results = {
            'data_type': data_type,
            'algorithm': algorithm_name,
            'num_contexts': num_contexts,
            'context_dim': context_dim,
            'num_actions': num_actions,
            'cumulative_regret': metrics['algo_0_cumulative_regret'],
            'avg_reward': metrics['algo_0_avg_reward'],
            'final_regret': metrics['algo_0_final_regret'],
            'data_time': data_time,
            'experiment_time': experiment_time,
            'total_time': data_time + experiment_time,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        print(f"  ✓ Completed in {results['total_time']:.2f}s")
        print(f"    Cumulative Regret: {results['cumulative_regret']:.4f}")
        print(f"    Average Reward: {results['avg_reward']:.4f}")
        
        return results, step_data
        
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
        return {
            'data_type': data_type,
            'algorithm': algorithm_name,
            'num_contexts': num_contexts,
            'context_dim': None,
            'num_actions': None,
            'cumulative_regret': None,
            'avg_reward': None,
            'final_regret': None,
            'data_time': None,
            'experiment_time': None,
            'total_time': None,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Run batch experiments for all algorithms and datasets')
    parser.add_argument('--num_contexts', type=int, default=1000,
                       help='Number of contexts to sample (default: 1000)')
    parser.add_argument('--data_dir', type=str, default='datasets',
                       help='Directory containing dataset files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--datasets', nargs='+', 
                       default=['linear', 'sparse_linear', 'wheel', 'mushroom', 'statlog', 'adult'],
                       help='Datasets to run (default: linear, sparse_linear, wheel, mushroom, statlog, adult)')
    parser.add_argument('--algorithms', nargs='+',
                       default=['neural_bandit', 'neural_linear', 'uniform', 'linear_full_posterior'],
                       help='Algorithms to run')
    
    args = parser.parse_args()
    
    # Create output directory
    # 链接输出路径
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define all available datasets and algorithms
    all_datasets = ['linear', 'sparse_linear', 'wheel', 'mushroom', 'financial', 'jester', 
                   'statlog', 'adult', 'covertype', 'census', 'statlog_shuttle', 'newsgroups']
    all_algorithms = ['neural_bandit', 'neural_linear', 'uniform', 'linear_full_posterior', 
                     'bootstrapped_bnn', 'fixed_policy',
                     'ucb1', 'linucb', 'epsilon_greedy', 'neural_ucb', 'neural_linucb']
    
    # Filter datasets and algorithms based on args
    datasets_to_run = [d for d in args.datasets if d in all_datasets]
    algorithms_to_run = [a for a in args.algorithms if a in all_algorithms]
    
    print(f"Running batch experiments:")
    print(f"  Datasets: {datasets_to_run}")
    print(f"  Algorithms: {algorithms_to_run}")
    print(f"  Number of contexts: {args.num_contexts}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    # Run experiments
    all_results = []
    total_experiments = len(datasets_to_run) * len(algorithms_to_run)
    completed_experiments = 0
    # 用于收集每个数据集的所有算法数据
    dataset_algorithm_data = {}
    
    start_time = time.time()
    
    for data_type in datasets_to_run:
        dataset_algorithm_data[data_type] = {}
        # 分别运行每个算法
        for algorithm_name in algorithms_to_run:
            completed_experiments += 1
            print(f"[{completed_experiments}/{total_experiments}] ", end="")
            # 传入数据类型、算法名称、上下文数量/轮数，数据目录、随机种子、输出目录
            # 返回统计结果和每轮奖励和遗憾
            results, step_data = run_single_experiment(
                data_type, algorithm_name, args.num_contexts, args.data_dir, args.seed, args.output_dir
            )
            #将统计结果拼接
            all_results.append(results)
            
             # 收集step数据用于合并保存
            if results['status'] == 'success':
                dataset_algorithm_data[data_type][algorithm_name] = step_data
            # Save intermediate results
            # 当运行了五个算法，则对数据进行一次存储
            if completed_experiments % 5 == 0:
                save_results(all_results, args.output_dir, num_contexts=args.num_contexts)
    
     # 为每个step数据集保存合并的CSV
        if dataset_algorithm_data[data_type]:
            save_combined_step_data(
                data_type, 
                dataset_algorithm_data[data_type], 
                args.output_dir, 
                args.seed,
                num_contexts=args.num_contexts
            )

    total_time = time.time() - start_time
    
    # Save final results
    save_results(all_results, args.output_dir, num_contexts=args.num_contexts)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {len([r for r in all_results if r['status'] == 'success'])}")
    print(f"Failed: {len([r for r in all_results if r['status'] == 'failed'])}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per experiment: {total_time/total_experiments:.2f} seconds")
    print(f"Results saved to: {args.output_dir}/")
    
    # Create summary table
    create_summary_table(all_results, args.output_dir)


def save_results(results: List[Dict], output_dir: str, num_contexts: int = 0):
    """Save results to CSV and JSON files with timestamp."""
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save as CSV
    csv_file = os.path.join(output_dir, f'{timestamp}_{num_contexts}r_results.csv')
    if results:
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)
    
    # Save as JSON
    json_file = os.path.join(output_dir, f'{timestamp}_{num_contexts}r_results.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def create_summary_table(results: List[Dict], output_dir: str):
    """Create a summary table of results with timestamp."""
    
    successful_results = [r for r in results if r['status'] == 'success']
    if not successful_results:
        print("No successful results to create summary table.")
        return
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create pivot table
    df = pd.DataFrame(successful_results)
    
    # Create regret table
    regret_table = df.pivot(index='data_type', columns='algorithm', values='cumulative_regret')
    # regret_file = os.path.join(output_dir, f'regret_summary_{timestamp}.csv')
    # regret_table.to_csv(regret_file)
    
    # Create reward table
    reward_table = df.pivot(index='data_type', columns='algorithm', values='avg_reward')
    # reward_file = os.path.join(output_dir, f'reward_summary_{timestamp}.csv')
    # reward_table.to_csv(reward_file)
    
    # print(f"\nSummary tables saved:")
    # print(f"  Regret summary: {regret_file}")
    # print(f"  Reward summary: {reward_file}")
    
    # Print summary tables
    print("\nCUMULATIVE REGRET SUMMARY:")
    print(regret_table.round(2))
    
    print("\nAVERAGE REWARD SUMMARY:")
    print(reward_table.round(4))


if __name__ == "__main__":
    main() 