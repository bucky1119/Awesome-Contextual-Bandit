#!/usr/bin/env python3
"""Plot cumulative regret and cumulative reward curves for all algorithms on linear and statlog datasets."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.dpi'] = 150

# --- Data files (latest run) ---
files = {
    'linear': {
        'regrets': 'results/detailed_logs/linear_regrets_seed42_20260302_150512.csv',
        'rewards': 'results/detailed_logs/linear_rewards_seed42_20260302_150512.csv',
    },
    'statlog': {
        'regrets': 'results/detailed_logs/statlog_regrets_seed42_20260302_150534.csv',
        'rewards': 'results/detailed_logs/statlog_rewards_seed42_20260302_150534.csv',
    },
}

# Algorithm display names and colors
ALGO_STYLE = {
    'neural_bandit':   {'label': 'Neural Bandit (greedy)',     'color': '#1f77b4', 'ls': '-'},
    'neural_linear':   {'label': 'Neural Linear TS (Bayesian)','color': '#ff7f0e', 'ls': '-'},
    'ucb1':            {'label': 'UCB1',                       'color': '#2ca02c', 'ls': '--'},
    'linucb':          {'label': 'LinUCB',                     'color': '#d62728', 'ls': '--'},
    'epsilon_greedy':  {'label': 'ε-greedy',                   'color': '#9467bd', 'ls': '--'},
    'neural_ucb':      {'label': 'Neural UCB',                 'color': '#8c564b', 'ls': '-.'},
    'neural_linucb':   {'label': 'Neural LinUCB',              'color': '#e377c2', 'ls': '-.'},
}

ALGO_ORDER = ['neural_bandit', 'neural_linear', 'ucb1', 'linucb',
              'epsilon_greedy', 'neural_ucb', 'neural_linucb']

os.makedirs('results/plots', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

for col_idx, dataset_name in enumerate(['linear', 'statlog']):
    df_reg = pd.read_csv(files[dataset_name]['regrets'])
    df_rwd = pd.read_csv(files[dataset_name]['rewards'])
    steps = df_reg['step'].values

    # --- Top row: Cumulative Regret ---
    ax_reg = axes[0, col_idx]
    for algo in ALGO_ORDER:
        col = f'{algo}_cumulative_regret'
        if col in df_reg.columns:
            style = ALGO_STYLE[algo]
            ax_reg.plot(steps, df_reg[col].values,
                        label=style['label'], color=style['color'],
                        linestyle=style['ls'], linewidth=1.8)
    ax_reg.set_title(f'{dataset_name.upper()} — Cumulative Regret', fontsize=14, fontweight='bold')
    ax_reg.set_xlabel('Round (t)')
    ax_reg.set_ylabel('Cumulative Regret')
    ax_reg.legend(fontsize=9, loc='upper left')
    ax_reg.grid(True, alpha=0.3)

    # --- Bottom row: Cumulative Reward ---
    ax_rwd = axes[1, col_idx]
    for algo in ALGO_ORDER:
        col = f'{algo}_cumulative_reward'
        if col in df_rwd.columns:
            style = ALGO_STYLE[algo]
            ax_rwd.plot(steps, df_rwd[col].values,
                        label=style['label'], color=style['color'],
                        linestyle=style['ls'], linewidth=1.8)
    # Also plot optimal cumulative reward
    if 'optimal_cumulative_reward' in df_rwd.columns:
        ax_rwd.plot(steps, df_rwd['optimal_cumulative_reward'].values,
                    label='Optimal', color='black', linestyle=':', linewidth=2.0)
    ax_rwd.set_title(f'{dataset_name.upper()} — Cumulative Reward', fontsize=14, fontweight='bold')
    ax_rwd.set_xlabel('Round (t)')
    ax_rwd.set_ylabel('Cumulative Reward')
    ax_rwd.legend(fontsize=9, loc='upper left')
    ax_rwd.grid(True, alpha=0.3)

plt.suptitle('Contextual Bandit Algorithm Comparison (2000 rounds, seed=42)',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()

out_path = 'results/plots/all_algorithms_comparison.png'
plt.savefig(out_path, bbox_inches='tight', dpi=150)
print(f'Saved to {out_path}')
plt.close()
