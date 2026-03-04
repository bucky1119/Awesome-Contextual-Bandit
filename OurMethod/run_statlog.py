#!/usr/bin/env python3
"""在 Statlog (Shuttle) 数据集上运行 OurMethod Combined UCB。

用法:
  python OurMethod/run_statlog.py                     # 默认 2000 轮
  python OurMethod/run_statlog.py --n_rounds 5000
  python OurMethod/run_statlog.py --n_rounds 5000 --seed 0 --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from OurMethod.core.protocol import Arm, Context, DecisionRecord
from OurMethod.core.prompt_builder import StructuredPromptBuilder
from OurMethod.core.frozen_llm import StubFrozenLLM
from OurMethod.core.compressor import MLPCompressor
from OurMethod.core.combined_policy import CombinedUCBPolicy
from OurMethod.core.online_runner import OnlineRunner


# ====================================================================== #
#  Statlog 数据集适配层                                                    #
# ====================================================================== #

def load_statlog(file_path: str, num_contexts: int = 0,
                 shuffle: bool = True, seed: int = 42):
    """加载 Statlog 数据集并转换为 bandit 格式。

    Returns:
        contexts:    (n, 9) 特征矩阵
        rewards:     (n, 7) 奖励矩阵 — 正确类别=1.0，其余=0.0
        opt_rewards: (n,) 最优奖励（全=1.0）
        opt_actions: (n,) 最优动作
    """
    data = np.loadtxt(file_path)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(data)
    if num_contexts > 0 and num_contexts < len(data):
        data = data[:num_contexts]

    contexts = data[:, :-1].astype(np.float64)
    labels = data[:, -1].astype(int) - 1          # 1-7 → 0-6

    num_actions = 7
    n = contexts.shape[0]
    rewards = np.zeros((n, num_actions))
    rewards[np.arange(n), labels] = 1.0
    opt_actions = labels
    opt_rewards = np.ones(n)
    return contexts, rewards, opt_rewards, opt_actions


class StatlogBanditEnv:
    """Statlog 数据集封装为在线 bandit 环境。

    核心问题：OnlineRunner 的 context_sampler / reward_fn / optimal_fn
    需要协调——同一轮中它们引用的必须是同一行。
    本类用游标维护当前行索引来解决这个问题。
    """

    def __init__(self, file_path: str, num_contexts: int = 0,
                 shuffle: bool = True, seed: int = 42,
                 normalize: bool = True):
        self.contexts, self.rewards, self.opt_rewards, self.opt_actions = \
            load_statlog(file_path, num_contexts, shuffle, seed)
        self.n, self.d = self.contexts.shape
        self.num_actions = self.rewards.shape[1]

        # 可选：特征标准化
        if normalize:
            mu = self.contexts.mean(axis=0)
            std = self.contexts.std(axis=0)
            std[std == 0] = 1.0
            self.contexts = (self.contexts - mu) / std

        self._cursor = -1

    def sample_context(self) -> np.ndarray:
        """返回下一行特征，循环使用数据。"""
        self._cursor = (self._cursor + 1) % self.n
        return self.contexts[self._cursor]

    def reward(self, features: np.ndarray, arm: int) -> float:
        """返回当前游标行对应 arm 的奖励。"""
        return float(self.rewards[self._cursor, arm])

    def optimal(self, features: np.ndarray) -> Tuple[float, int]:
        """返回 (最优奖励, 最优 arm)。"""
        return float(self.opt_rewards[self._cursor]), int(self.opt_actions[self._cursor])


# ====================================================================== #
#  主入口                                                                  #
# ====================================================================== #

def run_statlog_experiment(
    n_rounds: int = 2000,
    seed: int = 42,
    verbose: bool = True,
    hidden_dim: int = 64,
    z_dim: int = 16,
    c_ucb1: float = 1.0,
    alpha_linucb: float = 1.0,
    c_sim: float = 1.0,
):
    # ---- 1) 加载数据 ---- #
    data_path = os.path.join(ROOT, "datasets", "statlog.trn")
    if not os.path.exists(data_path):
        print(f"[错误] 找不到数据文件: {data_path}")
        return None

    env = StatlogBanditEnv(data_path, num_contexts=n_rounds, shuffle=True,
                           seed=seed, normalize=True)
    K = env.num_actions   # 7
    d = env.d             # 9

    print(f"Statlog 数据集: {env.n} 行, {d} 维特征, {K} 个动作")
    print(f"运行 {n_rounds} 轮, seed={seed}")

    # ---- 2) 构建 OurMethod 组件 ---- #
    arms = [Arm(arm_id=i, name=f"class_{i+1}",
                description=f"Statlog shuttle class {i+1}")
            for i in range(K)]

    pb = StructuredPromptBuilder()
    llm = StubFrozenLLM(hidden_dim=hidden_dim, generate_ds_llm=True)
    comp = MLPCompressor(input_dim=hidden_dim, output_dim=z_dim)
    policy = CombinedUCBPolicy(
        num_actions=K, z_dim=z_dim,
        c_ucb1=c_ucb1, alpha_linucb=alpha_linucb, c_sim=c_sim,
    )

    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"statlog_seed{seed}.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    runner = OnlineRunner(
        prompt_builder=pb,
        llm_encoder=llm,
        compressor=comp,
        policy=policy,
        arms=arms,
        env_reward_fn=env.reward,
        env_optimal_fn=env.optimal,
        context_sampler=env.sample_context,
        context_dim=d,
        log_path=log_path,
        seed=seed,
    )

    # ---- 3) 运行 ---- #
    t0 = time.time()
    history = runner.run(n_rounds, verbose=verbose)
    elapsed = time.time() - t0

    # ---- 4) 汇总结果 ---- #
    s = runner.summary()
    rewards = [r.reward for r in history]
    regrets = [r.regret for r in history if r.regret is not None]

    cum_regret = sum(regrets)
    avg_reward = np.mean(rewards)
    accuracy = np.mean([1.0 if r.reward == 1.0 else 0.0 for r in history])

    # 分段统计
    segment_size = max(n_rounds // 5, 1)
    print(f"\n{'='*60}")
    print(f"Statlog 实验结果  (耗时 {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  总轮次:         {n_rounds}")
    print(f"  累计奖励:       {sum(rewards):.1f} / {n_rounds}")
    print(f"  平均奖励:       {avg_reward:.4f}")
    print(f"  准确率:         {accuracy:.4f}")
    print(f"  累计遗憾:       {cum_regret:.1f}")
    print(f"\n  分段统计:")

    for seg in range(5):
        lo = seg * segment_size
        hi = min(lo + segment_size, n_rounds)
        seg_rewards = rewards[lo:hi]
        seg_regrets = regrets[lo:hi] if regrets else []
        seg_acc = np.mean([1.0 if r == 1.0 else 0.0 for r in seg_rewards])
        seg_avg = np.mean(seg_rewards) if seg_rewards else 0
        print(f"    [{lo:>5d}-{hi:>5d}]  avg_reward={seg_avg:.4f}  "
              f"accuracy={seg_acc:.4f}  "
              f"cum_regret={sum(seg_regrets):.1f}")

    # ---- UCB 主导信号来源分析 ---- #
    print(f"\n  UCB 约束来源分析 (最后 500 轮):")
    tail = history[-min(500, n_rounds):]
    dominant_counts = {"s1_linucb": 0, "s2_ucb1": 0, "s3_llm": 0}
    for rec in tail:
        debug = {}
        # Read from stored ucb lists
        if rec.ucb_values is not None and hasattr(policy, '_last_ucb_s1'):
            # we need to use policy's cached debug from the run
            pass
    # Re-analyze from logged data
    # Since we have policy debug only for last step, let's analyze from
    # the reward pattern
    print(f"    (详细 UCB 来源追踪需从 {log_path} 日志分析)")

    print(f"\n  日志: {log_path}")
    return history


def main():
    p = argparse.ArgumentParser(description="OurMethod on Statlog dataset")
    p.add_argument("--n_rounds", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--z_dim", type=int, default=16)
    p.add_argument("--c_ucb1", type=float, default=1.0)
    p.add_argument("--alpha_linucb", type=float, default=1.0)
    p.add_argument("--c_sim", type=float, default=1.0)
    args = p.parse_args()
    run_statlog_experiment(**vars(args))


if __name__ == "__main__":
    main()
