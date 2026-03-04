#!/usr/bin/env python3
"""
统一对比实验：在 Statlog 数据集上运行 OurMethod 与 baseline 算法并绘图。

算法：
  1. neural_bandit    — 神经网络直接输出策略（epsilon-greedy）
  2. neural_linear    — Neural Linear Posterior Sampling
  3. combined_ucb     — OurMethod: min{UCB^s1, UCB^s2, UCB^s3}

用法:
  python OurMethod/compare_statlog.py
  python OurMethod/compare_statlog.py --n_rounds 5000 --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- baseline 算法 ---- #
from bandits.algorithms.neural_bandit_model import NeuralBanditModel
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.core.contextual_bandit import ContextualBandit

# ---- OurMethod ---- #
from OurMethod.core.protocol import Arm, Context, DecisionRecord
from OurMethod.core.prompt_builder import StructuredPromptBuilder
from OurMethod.core.frozen_llm import StubFrozenLLM
from OurMethod.core.compressor import MLPCompressor
from OurMethod.core.combined_policy import CombinedUCBPolicy

# ---- 数据采样 ---- #
from bandits.data.data_sampler import sample_statlog_data

# ================================================================== #
#  输出目录                                                           #
# ================================================================== #
RESULTS_DIR = os.path.join(ROOT, "results")
DETAILED_DIR = os.path.join(RESULTS_DIR, "detailed_logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

for d in [RESULTS_DIR, DETAILED_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ================================================================== #
#  baseline 运行器：与 run_all_experiments.py 完全一致                  #
# ================================================================== #

def create_baseline(name: str, num_actions: int, context_dim: int) -> Any:
    if name == "neural_bandit":
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
        }
        return NeuralBanditModel(hparams, name="neural_bandit")
    elif name == "neural_linear":
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
            "lambda_prior": 0.25,
            "a0": 6,
            "b0": 6,
            "training_freq": 100,
            "training_freq_network": 100,
            "training_epochs": 100,
            "initial_pulls": 2,
        }
        return NeuralLinearPosteriorSampling(hparams, name="neural_linear")
    else:
        raise ValueError(f"Unknown baseline: {name}")


def run_baseline(
    algo,
    algo_name: str,
    cmab: ContextualBandit,
    opt_rewards: np.ndarray,
    opt_actions: np.ndarray,
    n_rounds: int,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """逐步运行 baseline 并返回每步结果。"""
    actions = np.zeros(n_rounds, dtype=np.int64)
    rewards = np.zeros(n_rounds)

    t0 = time.time()
    for i in range(n_rounds):
        ctx = cmab.context(i)
        a = algo.action(ctx)
        r = cmab.reward(i, a)
        algo.update(ctx, a, r)
        actions[i] = a
        rewards[i] = r

        if verbose and i % max(1, n_rounds // 10) == 0:
            cr = np.sum(opt_rewards[: i + 1] - rewards[: i + 1])
            print(f"  [{algo_name}] step {i:>5d}  cum_regret={cr:.1f}")

    elapsed = time.time() - t0
    step_regret = opt_rewards[:n_rounds] - rewards
    cum_regret = np.cumsum(step_regret)
    print(f"  [{algo_name}] 完成 — cum_regret={cum_regret[-1]:.1f}  "
          f"avg_reward={np.mean(rewards):.4f}  耗时={elapsed:.1f}s")
    return {
        "actions": actions,
        "rewards": rewards,
        "opt_rewards": opt_rewards[:n_rounds],
        "opt_actions": opt_actions[:n_rounds],
        "step_regret": step_regret,
        "cumulative_regret": cum_regret,
        "cumulative_reward": np.cumsum(rewards),
    }


# ================================================================== #
#  OurMethod 运行器                                                   #
# ================================================================== #

def run_our_method(
    cmab: ContextualBandit,
    opt_rewards: np.ndarray,
    opt_actions: np.ndarray,
    n_rounds: int,
    num_actions: int,
    context_dim: int,
    seed: int = 42,
    hidden_dim: int = 64,
    z_dim: int = 16,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """逐步运行 OurMethod (Combined UCB)。"""

    arms = [Arm(arm_id=i, name=f"class_{i + 1}") for i in range(num_actions)]
    pb = StructuredPromptBuilder()
    llm = StubFrozenLLM(hidden_dim=hidden_dim, generate_ds_llm=True)
    comp = MLPCompressor(input_dim=hidden_dim, output_dim=z_dim)
    policy = CombinedUCBPolicy(num_actions=num_actions, z_dim=z_dim)

    rng = np.random.RandomState(seed)
    actions = np.zeros(n_rounds, dtype=np.int64)
    rewards = np.zeros(n_rounds)
    history: List[DecisionRecord] = []

    t0 = time.time()
    for i in range(n_rounds):
        feat = cmab.context(i)
        ctx = Context(features=feat)

        # prompt → LLM → compress → select
        feedback = history[-10:] if history else None
        prompt = pb.build(ctx, arms, feedback)
        h_t, ds_llm = llm.encode(prompt, num_arms=num_actions)
        z_t = comp.forward(h_t)

        arm_id = policy.select(ctx, arms, z_t, ds_llm)
        r = cmab.reward(i, arm_id)

        rec = DecisionRecord(
            step=i,
            context=ctx,
            arms=arms,
            prompt=prompt,
            h_t=h_t,
            z_t=z_t,
            ds_llm=ds_llm,
            chosen_arm=arm_id,
            reward=r,
            optimal_reward=float(opt_rewards[i]),
        )
        policy.update(rec)
        history.append(rec)

        actions[i] = arm_id
        rewards[i] = r

        if verbose and i % max(1, n_rounds // 10) == 0:
            cr = np.sum(opt_rewards[: i + 1] - rewards[: i + 1])
            print(f"  [combined_ucb] step {i:>5d}  cum_regret={cr:.1f}")

    elapsed = time.time() - t0
    step_regret = opt_rewards[:n_rounds] - rewards
    cum_regret = np.cumsum(step_regret)
    print(f"  [combined_ucb] 完成 — cum_regret={cum_regret[-1]:.1f}  "
          f"avg_reward={np.mean(rewards):.4f}  耗时={elapsed:.1f}s")
    return {
        "actions": actions,
        "rewards": rewards,
        "opt_rewards": opt_rewards[:n_rounds],
        "opt_actions": opt_actions[:n_rounds],
        "step_regret": step_regret,
        "cumulative_regret": cum_regret,
        "cumulative_reward": np.cumsum(rewards),
    }


# ================================================================== #
#  保存日志（兼容 visualize_results.py 格式）                          #
# ================================================================== #

def save_logs(
    all_data: Dict[str, Dict[str, np.ndarray]],
    n_rounds: int,
    seed: int,
    timestamp: str,
):
    """保存每个算法的单独 CSV 和合并的 regrets / rewards CSV。"""

    # ---- 1) 每个算法单独 CSV ---- #
    for algo_name, data in all_data.items():
        filepath = os.path.join(
            DETAILED_DIR, f"statlog_{algo_name}_seed{seed}_{timestamp}.csv"
        )
        df = pd.DataFrame(
            {
                "step": np.arange(n_rounds),
                "action": data["actions"],
                "reward": data["rewards"],
                "opt_action": data["opt_actions"],
                "opt_reward": data["opt_rewards"],
                "is_optimal": (data["actions"] == data["opt_actions"]).astype(int),
                "step_regret": data["step_regret"],
                "cumulative_regret": data["cumulative_regret"],
                "cumulative_reward": data["cumulative_reward"],
            }
        )
        df.to_csv(filepath, index=False)
        print(f"  保存: {filepath}")

    # ---- 2) 合并 rewards CSV ---- #
    reward_path = os.path.join(
        DETAILED_DIR, f"statlog_rewards_seed{seed}_{timestamp}.csv"
    )
    reward_df = {"step": np.arange(n_rounds)}
    for name, data in all_data.items():
        reward_df[f"{name}_reward"] = data["rewards"]
        reward_df[f"{name}_cumulative_reward"] = data["cumulative_reward"]
    first = next(iter(all_data.values()))
    reward_df["optimal_reward"] = first["opt_rewards"]
    reward_df["optimal_cumulative_reward"] = np.cumsum(first["opt_rewards"])
    pd.DataFrame(reward_df).to_csv(reward_path, index=False)
    print(f"  保存合并奖励: {reward_path}")

    # ---- 3) 合并 regrets CSV ---- #
    regret_path = os.path.join(
        DETAILED_DIR, f"statlog_regrets_seed{seed}_{timestamp}.csv"
    )
    regret_df = {"step": np.arange(n_rounds)}
    for name, data in all_data.items():
        regret_df[f"{name}_step_regret"] = data["step_regret"]
        regret_df[f"{name}_cumulative_regret"] = data["cumulative_regret"]
    regret_df["optimal_reward"] = first["opt_rewards"]
    pd.DataFrame(regret_df).to_csv(regret_path, index=False)
    print(f"  保存合并遗憾: {regret_path}")


# ================================================================== #
#  绘图                                                               #
# ================================================================== #

def plot_curves(
    all_data: Dict[str, Dict[str, np.ndarray]],
    n_rounds: int,
    seed: int,
    timestamp: str,
):
    """绘制累积遗憾曲线和累积奖励曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    steps = np.arange(n_rounds)

    # ---- 颜色 / 线型 ---- #
    styles = {
        "neural_bandit": {"color": "#1f77b4", "ls": "--", "lw": 2},
        "neural_linear": {"color": "#ff7f0e", "ls": "-.", "lw": 2},
        "combined_ucb":  {"color": "#2ca02c", "ls": "-",  "lw": 2.5},
    }

    # ========== 累积遗憾曲线 ========== #
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in all_data.items():
        s = styles.get(name, {"color": "gray", "ls": "-", "lw": 1})
        ax.plot(steps, data["cumulative_regret"], label=name, **s)
    ax.set_xlabel("Time Step", fontsize=13)
    ax.set_ylabel("Cumulative Regret", fontsize=13)
    ax.set_title(f"Statlog — Cumulative Regret Comparison (seed={seed})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    regret_png = os.path.join(
        PLOTS_DIR, f"statlog_cumulative_regret_seed{seed}_{timestamp}.png"
    )
    fig.savefig(regret_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存遗憾曲线: {regret_png}")

    # ========== 累积奖励曲线 ========== #
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in all_data.items():
        s = styles.get(name, {"color": "gray", "ls": "-", "lw": 1})
        ax.plot(steps, data["cumulative_reward"], label=name, **s)
    # 最优基线
    opt_cum = np.cumsum(next(iter(all_data.values()))["opt_rewards"])
    ax.plot(steps, opt_cum, label="optimal", color="red", ls=":", lw=2)
    ax.set_xlabel("Time Step", fontsize=13)
    ax.set_ylabel("Cumulative Reward", fontsize=13)
    ax.set_title(f"Statlog — Cumulative Reward Comparison (seed={seed})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    reward_png = os.path.join(
        PLOTS_DIR, f"statlog_cumulative_reward_seed{seed}_{timestamp}.png"
    )
    fig.savefig(reward_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存奖励曲线: {reward_png}")

    # ========== 滑动窗口平均奖励曲线 ========== #
    fig, ax = plt.subplots(figsize=(12, 6))
    window = min(200, n_rounds // 5)
    for name, data in all_data.items():
        s = styles.get(name, {"color": "gray", "ls": "-", "lw": 1})
        rolling_avg = pd.Series(data["rewards"]).rolling(window, min_periods=1).mean()
        ax.plot(steps, rolling_avg, label=name, **s)
    ax.axhline(y=1.0, color="red", ls=":", lw=1.5, label="optimal (=1.0)")
    ax.set_xlabel("Time Step", fontsize=13)
    ax.set_ylabel(f"Average Reward (window={window})", fontsize=13)
    ax.set_title(f"Statlog — Rolling Average Reward (seed={seed})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    rolling_png = os.path.join(
        PLOTS_DIR, f"statlog_rolling_reward_seed{seed}_{timestamp}.png"
    )
    fig.savefig(rolling_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存滑动奖励曲线: {rolling_png}")


# ================================================================== #
#  汇总表                                                             #
# ================================================================== #

def print_summary(all_data: Dict[str, Dict[str, np.ndarray]], n_rounds: int):
    print(f"\n{'='*70}")
    print(f"{'算法':<20s} {'累计遗憾':>12s} {'平均奖励':>12s} {'准确率':>10s}")
    print(f"{'='*70}")
    for name, data in all_data.items():
        cr = data["cumulative_regret"][-1]
        avg_r = np.mean(data["rewards"])
        acc = np.mean(data["actions"] == data["opt_actions"])
        print(f"{name:<20s} {cr:>12.1f} {avg_r:>12.4f} {acc:>10.4f}")
    print(f"{'='*70}")

    # 分段对比
    seg = max(n_rounds // 5, 1)
    print(f"\n分段平均奖励 (每段 {seg} 轮):")
    header = f"{'区间':<16s}"
    for name in all_data:
        header += f" {name:>16s}"
    print(header)
    print("-" * len(header))
    for s in range(5):
        lo, hi = s * seg, min((s + 1) * seg, n_rounds)
        row = f"[{lo:>5d}-{hi:>5d}]    "
        for name, data in all_data.items():
            avg = np.mean(data["rewards"][lo:hi])
            row += f" {avg:>16.4f}"
        print(row)


# ================================================================== #
#  主入口                                                             #
# ================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="OurMethod vs baseline on Statlog"
    )
    parser.add_argument("--n_rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    n_rounds = args.n_rounds
    seed = args.seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    np.random.seed(seed)

    # ---- 1) 采样 Statlog 数据 ---- #
    data_path = os.path.join(ROOT, "datasets", "statlog.trn")
    if not os.path.exists(data_path):
        print(f"[错误] 找不到 {data_path}")
        sys.exit(1)

    n_rounds = min(n_rounds, 43500)
    dataset, (opt_rewards, opt_actions) = sample_statlog_data(
        data_path, n_rounds, shuffle_rows=True
    )
    num_actions = 7
    context_dim = dataset.shape[1] - num_actions

    print(f"Statlog 数据: {n_rounds} 行, {context_dim} 维特征, {num_actions} 个动作")
    print(f"seed={seed}, timestamp={timestamp}")
    print()

    # 创建共享的 ContextualBandit 环境
    cmab = ContextualBandit(num_actions, context_dim)
    cmab.feed_data(dataset)

    # ---- 2) 运行各算法 ---- #
    all_data: Dict[str, Dict[str, np.ndarray]] = {}

    # ---- neural_bandit ---- #
    print("=" * 60)
    print("运行 neural_bandit ...")
    np.random.seed(seed)
    algo_nb = create_baseline("neural_bandit", num_actions, context_dim)
    all_data["neural_bandit"] = run_baseline(
        algo_nb, "neural_bandit", cmab, opt_rewards, opt_actions,
        n_rounds, verbose=args.verbose,
    )

    # ---- neural_linear ---- #
    print("\n" + "=" * 60)
    print("运行 neural_linear ...")
    np.random.seed(seed)
    algo_nl = create_baseline("neural_linear", num_actions, context_dim)
    all_data["neural_linear"] = run_baseline(
        algo_nl, "neural_linear", cmab, opt_rewards, opt_actions,
        n_rounds, verbose=args.verbose,
    )

    # ---- combined_ucb (OurMethod) ---- #
    print("\n" + "=" * 60)
    print("运行 combined_ucb (OurMethod) ...")
    np.random.seed(seed)
    all_data["combined_ucb"] = run_our_method(
        cmab, opt_rewards, opt_actions, n_rounds,
        num_actions, context_dim,
        seed=seed,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        verbose=args.verbose,
    )

    # ---- 3) 保存日志 ---- #
    print("\n" + "=" * 60)
    print("保存日志 ...")
    save_logs(all_data, n_rounds, seed, timestamp)

    # ---- 4) 汇总 ---- #
    print_summary(all_data, n_rounds)

    # ---- 5) 绘图 ---- #
    print("\n绘制曲线 ...")
    plot_curves(all_data, n_rounds, seed, timestamp)

    print(f"\n✅ 全部完成!  日志与图表保存在 {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
