#!/usr/bin/env python3
"""
在 statlog / magic / covertype / mnist 数据集上运行 InterNeuralBandit 外部项目的全部算法并绘图。

算法列表（来自 InterNeuralBandit/ 目录）：
  lin_ucb         — 线性 UCB（per-arm disjoint，arm one-hot 编码）
  lin_ts          — 线性 Thompson Sampling
  lin_ucb_rp      — 随机映射（RP）+ LinUCB
  exp3            — 上下文无关 EXP3 bandit
  inter_neural_ucb — InterNeural UCB（交替在线 LinUCB + 离线 NN 训练）
  inter_neural_ts  — InterNeural TS（交替在线 LinTS  + 离线 NN 训练）
  neural_ucb      — Neural UCB（基于梯度的全参数探索，批量运行）

使用方式：
  # 跑所有数据集（默认 n_rounds=1000）
  python run_inter_neural.py --datasets statlog magic covertype mnist --n_rounds 1000 --seed 42

  # 仅绘图（复用已缓存的 .npz 结果）
  python run_inter_neural.py --datasets statlog --n_rounds 1000 --seed 42 --plot_only

  # 跳过耗时的 neural_ucb，只跑其余算法
  python run_inter_neural.py --datasets statlog --n_rounds 1000 --skip neural_ucb
"""

from __future__ import annotations

import sys
import os
import time
import math
import random as _random
import argparse
import json
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn import random_projection as _sk_rp

# ─── 路径设置 ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bandits.data.data_sampler import (
    sample_statlog_data,
    sample_magic_data,
    sample_covertype_data,
    sample_mnist_data,
)

# InterNeuralBandit 的神经网络模型（仅依赖 torch，无额外依赖）
from InterNeuralBandit.model import Model as _INNModel

# NeuralUCB 与 ContextualBandit（neural_ucb.py 自包含，无外部 base 依赖）
from InterNeuralBandit.algorithms.neural_ucb import (
    NeuralUCB as _ExtNeuralUCB,
    ContextualBandit as _ExtCBandit,
)

# ─── 结果目录 ──────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(ROOT, "results", "inter_neural_bandit")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── 随机种子 ──────────────────────────────────────────────────────────────────

def _set_seeds(seed: int):
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ─── 轻量 Action / Reward shim（替代外部 base 包）────────────────────────────

class _Action:
    def __init__(self, actions=None, k: int = 1):
        if isinstance(actions, np.ndarray):
            self.actions = actions.copy()
        elif isinstance(actions, (int, np.integer)):
            self.actions = np.array([int(actions)])
        else:
            self.actions = np.arange(k)

class _Reward:
    def __init__(self, rewards=None, k: int = 1):
        if isinstance(rewards, np.ndarray):
            self.rewards = rewards.copy().astype(float)
        elif isinstance(rewards, (int, float, np.floating)):
            self.rewards = np.array([float(rewards)])
        else:
            self.rewards = np.zeros(k)

# ─────────────────────────────────────────────────────────────────────────────
#  Algorithm 1: LinUCB — 逐 arm disjoint 线性 UCB
#  对应 InterNeuralBandit/algorithms/ucb.py 中的 LinearPayoffUCB，
#  但采用 per-arm 模型适配共享上下文数据集。
# ─────────────────────────────────────────────────────────────────────────────

class LinUCB:
    """LinUCB with disjoint per-arm ridge regression + UCB exploration."""

    def __init__(self, context_dim: int, n_arms: int,
                 alpha: float = 0.3, lambda_: float = 1.0):
        self.n_arms = n_arms
        self.d = context_dim
        self.alpha = alpha
        # per-arm matrices
        self.A_inv = [(1.0 / lambda_) * np.eye(self.d) for _ in range(n_arms)]
        self.b = [np.zeros(self.d) for _ in range(n_arms)]
        self.theta = [np.zeros(self.d) for _ in range(n_arms)]

    def action(self, context: np.ndarray) -> int:
        ucb = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            pred = float(np.dot(self.theta[a], context))
            conf = self.alpha * float(np.sqrt(max(0.0, np.dot(context, self.A_inv[a] @ context))))
            ucb[a] = pred + conf
        return int(np.argmax(ucb))

    def update(self, context: np.ndarray, arm: int, reward: float):
        Ax = self.A_inv[arm] @ context
        denom = 1.0 + float(context @ Ax)
        self.A_inv[arm] -= np.outer(Ax, Ax) / denom
        self.b[arm] += reward * context
        self.theta[arm] = self.A_inv[arm] @ self.b[arm]

# ─────────────────────────────────────────────────────────────────────────────
#  Algorithm 2: LinTS — 线性 Thompson Sampling
#  对应 LinearPayoffTS（per-arm 版本）
# ─────────────────────────────────────────────────────────────────────────────

class LinTS:
    """Thompson Sampling with Gaussian linear posterior (per-arm)."""

    def __init__(self, context_dim: int, n_arms: int,
                 var: float = 0.3, lambda_: float = 1.0):
        self.n_arms = n_arms
        self.d = context_dim
        self.var = var
        self.A_inv = [(1.0 / lambda_) * np.eye(self.d) for _ in range(n_arms)]
        self.b = [np.zeros(self.d) for _ in range(n_arms)]
        self.theta = [np.zeros(self.d) for _ in range(n_arms)]

    def action(self, context: np.ndarray) -> int:
        vals = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            try:
                # 数值稳定：确保协方差矩阵半正定
                cov = self.var * self.A_inv[a]
                cov = 0.5 * (cov + cov.T)
                w, v = np.linalg.eigh(cov)
                w = np.maximum(w, 1e-10)
                cov_psd = (v * w) @ v.T
                theta_s = np.random.multivariate_normal(self.theta[a], cov_psd)
            except Exception:
                theta_s = self.theta[a]
            vals[a] = float(theta_s @ context)
        return int(np.argmax(vals))

    def update(self, context: np.ndarray, arm: int, reward: float):
        Ax = self.A_inv[arm] @ context
        denom = 1.0 + float(context @ Ax)
        self.A_inv[arm] -= np.outer(Ax, Ax) / denom
        self.b[arm] += reward * context
        self.theta[arm] = self.A_inv[arm] @ self.b[arm]

# ─────────────────────────────────────────────────────────────────────────────
#  Algorithm 3: LinUCB with Random Projection
#  对应 LinearPayoffUCBWithRP — 将高维上下文随机映射到低维后运行共享 LinUCB
# ─────────────────────────────────────────────────────────────────────────────

class LinUCBRP:
    """LinUCB with Gaussian Random Projection. Shared model on projected features."""

    def __init__(self, context_dim: int, n_arms: int,
                 rp_dim: int = 10, alpha: float = 0.3, lambda_: float = 1.0):
        self.n_arms = n_arms
        self.alpha = alpha
        self.rp_dim = min(rp_dim, context_dim)
        # 生成 Gaussian RP 矩阵: (context_dim, rp_dim)
        transformer = _sk_rp.GaussianRandomProjection(
            n_components=self.rp_dim, random_state=42
        )
        dummy = np.random.randn(1, context_dim)
        self.RP = transformer.fit(dummy).components_.T  # (context_dim, rp_dim)

        # 共享矩阵（Shared LinUCB on projected features）
        d = self.rp_dim
        self.A_inv = (1.0 / lambda_) * np.eye(d)
        self.b = np.zeros(d)
        self.theta = np.zeros(d)

    def _project(self, context: np.ndarray) -> np.ndarray:
        return context @ self.RP  # (rp_dim,)

    def action(self, context: np.ndarray) -> int:
        # 对所有 arm 用同一个投影后的特征（共享上下文），
        # 直接评估 UCB 取 argmax（arm 不可区分时退化为随机探索，添加随机 tie-breaking）
        phi = self._project(context)
        pred = float(phi @ self.theta)
        conf = self.alpha * float(np.sqrt(max(0.0, phi @ self.A_inv @ phi)))
        # 所有 arm 得分相同时随机选臂；否则取最高 UCB
        # 注：若要完整使用 RP 区分 arm，需添加 arm one-hot（见 InterNeural 系列算法）
        _unused = pred + conf  # noqa
        return int(np.random.randint(0, self.n_arms)) if self.n_arms == 1 else self._pick_arm_with_rp(context)

    def _pick_arm_with_rp(self, context: np.ndarray) -> int:
        """在 RP 基础上添加 arm one-hot 向量以区分不同臂。"""
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            # 将 arm one-hot 拼接到原始 context（在投影前），但投影矩阵维度固定
            # 改为：在投影后特征上拼接 arm_id 的 sin/cos 编码（保持 rp_dim 不变）
            # 简单处理：per-arm UCB，各自存 A_inv 和 b
            scores[a] = float(self._arm_theta[a] @ context)
            conf = self.alpha * float(np.sqrt(max(0.0, context @ self._arm_Ainv[a] @ context)))
            scores[a] += conf
        return int(np.argmax(scores))

    # 内部重构为 per-arm RP LinUCB（固定投影，per-arm 模型）
    def __init__(self, context_dim: int, n_arms: int,  # noqa: F811
                 rp_dim: int = 10, alpha: float = 0.3, lambda_: float = 1.0):
        self.n_arms = n_arms
        self.alpha = alpha
        self.rp_dim = min(rp_dim, context_dim)
        transformer = _sk_rp.GaussianRandomProjection(
            n_components=self.rp_dim, random_state=42
        )
        dummy = np.random.randn(max(2, self.rp_dim + 1), context_dim)
        self.RP = transformer.fit(dummy).components_.T  # (context_dim, rp_dim)

        d = self.rp_dim
        self._arm_Ainv = [(1.0 / lambda_) * np.eye(d) for _ in range(n_arms)]
        self._arm_b = [np.zeros(d) for _ in range(n_arms)]
        self._arm_theta = [np.zeros(d) for _ in range(n_arms)]

    def action(self, context: np.ndarray) -> int:
        phi = context @ self.RP  # (rp_dim,)
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            pred = float(phi @ self._arm_theta[a])
            conf = self.alpha * float(np.sqrt(max(0.0, phi @ self._arm_Ainv[a] @ phi)))
            scores[a] = pred + conf
        return int(np.argmax(scores))

    def update(self, context: np.ndarray, arm: int, reward: float):
        phi = context @ self.RP
        Ax = self._arm_Ainv[arm] @ phi
        denom = 1.0 + float(phi @ Ax)
        self._arm_Ainv[arm] -= np.outer(Ax, Ax) / denom
        self._arm_b[arm] += reward * phi
        self._arm_theta[arm] = self._arm_Ainv[arm] @ self._arm_b[arm]

# ─────────────────────────────────────────────────────────────────────────────
#  Algorithm 4: EXP3 — 上下文无关的 EXP3 bandit
#  直接根据 InterNeuralBandit/algorithms/exp3.py 逻辑实现
# ─────────────────────────────────────────────────────────────────────────────

class Exp3:
    """EXP3 context-free bandit (importance-weighted exponential update)."""

    def __init__(self, n_arms: int, gamma: float = 0.1):
        self.n_arms = n_arms
        self.gamma = gamma
        self.weights = np.ones(n_arms, dtype=np.float64)
        self._p = np.ones(n_arms) / n_arms

    def action(self, _context=None) -> int:
        weight_sum = float(np.sum(self.weights))
        self._p = (
            (1.0 - self.gamma) * self.weights / weight_sum
            + self.gamma / self.n_arms
        )
        # 按概率 p 采样
        return int(np.random.choice(self.n_arms, p=self._p))

    def update(self, _context, arm: int, reward: float):
        r_hat = reward / max(self._p[arm], 1e-12)  # importance-weighted estimate
        self.weights[arm] *= math.exp(
            r_hat * self.gamma / self.n_arms
        )
        # 防止数值溢出
        self.weights /= np.max(self.weights)

# ─────────────────────────────────────────────────────────────────────────────
#  共用辅助：arm one-hot 编码上下文矩阵构建
# ─────────────────────────────────────────────────────────────────────────────

def _build_aug_contexts(context: np.ndarray, n_arms: int) -> np.ndarray:
    """将共享 context 与 arm one-hot 拼接，返回 (n_arms, context_dim + n_arms)。

    这使得只有共享特征的数据集也能让 InterNeural 的共享线性模型区分不同臂。
    """
    d = len(context)
    cmat = np.zeros((n_arms, d + n_arms), dtype=np.float32)
    for a in range(n_arms):
        cmat[a, :d] = context
        cmat[a, d + a] = 1.0
    return cmat

# ─────────────────────────────────────────────────────────────────────────────
#  Algorithm 5 & 6: InterNeuralUCB / InterNeuralTS
#
#  核心思想（来自论文）：交替固定点优化
#    - 在线阶段：固定 NN，用 LinUCB/LinTS 在潜在特征上选臂（更新线性参数 θ）
#    - 离线阶段：固定线性参数 θ，重训 NN（更新神经网络权重 f）
#
#  适配我们的数据集（共享上下文）：
#    - context 通过 arm one-hot 编码扩展，使 NN 能为不同 arm 生成不同潜在表示
#    - 历史缓冲区存储 [reward, context_aug...] 格式
# ─────────────────────────────────────────────────────────────────────────────

class _InterNeuralBase:
    """InterNeuralUCB / InterNeuralTS 公共基类。"""

    def __init__(
        self,
        context_dim: int, n_arms: int,
        latent_dim: int = 10,
        offline_lr: float = 1e-3,
        offline_epochs: int = 50,
        online_max_step: int = 50,
        online_batch_size: int = 50,
        historical_data_size: int = 100,
    ):
        self.n_arms = n_arms
        self.aug_dim = context_dim + n_arms   # NN 输入维度（原始特征 + arm one-hot）
        self.latent_dim = latent_dim
        self.offline_lr = offline_lr
        self.offline_epochs = offline_epochs
        self.online_max_step = online_max_step
        self.online_batch_size = online_batch_size
        self.historical_data_size = historical_data_size

        # 神经网络：input→input→input→input→latent（来自外部项目 model.py）
        self.model = _INNModel(self.aug_dim, latent_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=offline_lr)
        self.criterion = nn.MSELoss()

        # 共享线性模型（作用于所有 arm 的潜在特征）
        self._lambda = 1.0
        self._init_linear()

        # 历史缓冲区：形状 (N, 1 + aug_dim)，第 0 列为 reward
        # 用随机小值初始化，防止 warm-up 时为空
        rng = np.random.default_rng(0)
        self._history = rng.standard_normal((historical_data_size, 1 + self.aug_dim))
        self._history[:, 0] = 0.0   # reward 初始化为 0

        # 迭代计数器
        self._iter_step = 0    # 当前迭代内已走的真实步数
        self._total_steps = 0  # 全局步数

    def _init_linear(self):
        """重置共享线性模型（每次迭代开始时调用）。"""
        d = self.latent_dim
        self._A = self._lambda * np.eye(d)
        self._A_inv = (1.0 / self._lambda) * np.eye(d)
        self._b = np.zeros(d)
        self._linpara = np.zeros(d)

    def _get_latent(self, cmat: np.ndarray) -> np.ndarray:
        """(n_arms, aug_dim) → (n_arms, latent_dim)，eval 模式无梯度。"""
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.FloatTensor(cmat)).numpy()

    def _update_linear(self, phi: np.ndarray, reward: float):
        """Sherman-Morrison 增量更新共享 A_inv、b、linpara。"""
        Ax = self._A_inv @ phi
        denom = 1.0 + float(phi @ Ax)
        self._A_inv -= np.outer(Ax, Ax) / denom
        self._A += np.outer(phi, phi)
        self._b += reward * phi
        self._linpara = self._A_inv @ self._b

    def _warmup(self):
        """用历史缓冲区热启动线性模型（固定 NN 权重）。"""
        n = len(self._history)
        k = min(self.online_batch_size, n)
        idx = np.random.choice(n, k, replace=False)
        for i in idx:
            row = self._history[i]
            cxt_aug = row[1:].reshape(1, -1).astype(np.float32)
            phi = self._get_latent(cxt_aug).flatten()
            self._update_linear(phi, float(row[0]))

    def _offline_train(self):
        """用历史缓冲区重训 NN（固定 linpara）。"""
        n = len(self._history)
        k = min(self.historical_data_size, n)
        idx = np.random.choice(n, k, replace=True)
        features = torch.FloatTensor(self._history[idx, 1:])   # (k, aug_dim)
        rewards_t = torch.FloatTensor(self._history[idx, 0])   # (k,)
        linpara_t = torch.tensor(self._linpara, dtype=torch.float32)

        self.model.train()
        for _ in range(self.offline_epochs):
            self.optimizer.zero_grad()
            latent = self.model(features)                 # (k, latent_dim)
            pred = latent.matmul(linpara_t)               # (k,)
            loss = self.criterion(pred, rewards_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

    def _select_arm(self, latent: np.ndarray) -> int:
        """子类实现 UCB / TS。"""
        raise NotImplementedError

    def step(self, context: np.ndarray, rewards_all: np.ndarray) -> int:
        """标准接口：输入共享 context 和所有臂的奖励，返回选中臂编号。

        内部自动管理：在线 → 离线切换、线性模型重初始化、历史缓冲区更新。
        """
        # 迭代开始：（第一步或上一轮结束后）初始化线性模型 + 热启动
        if self._iter_step == 0:
            if self._total_steps > 0:    # 非第一次迭代，先做离线训练
                self._offline_train()
            self._init_linear()
            self._warmup()

        # 构建 arm-augmented 上下文矩阵
        cmat = _build_aug_contexts(context, self.n_arms)   # (n_arms, aug_dim)

        # 获取潜在表示
        latent = self._get_latent(cmat)                    # (n_arms, latent_dim)

        # 选臂
        arm = self._select_arm(latent)

        # 观察奖励，更新共享线性模型
        phi = latent[arm]
        reward = float(rewards_all[arm])
        self._update_linear(phi, reward)

        # 追加到历史缓冲区 [reward, context_aug_chosen]
        row = np.concatenate([[reward], cmat[arm]]).reshape(1, -1)
        self._history = np.vstack([self._history, row])

        # 更新计数器
        self._iter_step = (self._iter_step + 1) % self.online_max_step
        self._total_steps += 1

        return arm

    # 统一接口（与其他算法一致，runner 根据 has_step 属性选择调用方式）
    has_step = True

    def action(self, context: np.ndarray) -> int:
        """占位符，InterNeural 应使用 step()。"""
        raise RuntimeError("Please call step(context, rewards_all) for InterNeural algorithms.")

    def update(self, context: np.ndarray, arm: int, reward: float):
        raise RuntimeError("Please call step(context, rewards_all) for InterNeural algorithms.")


class InterNeuralUCB(_InterNeuralBase):
    """InterNeuralUCB — 在线阶段用 LinUCB 选臂。"""

    def __init__(self, context_dim: int, n_arms: int, alpha: float = 1.0, **kwargs):
        super().__init__(context_dim, n_arms, **kwargs)
        self.alpha = alpha

    def _select_arm(self, latent: np.ndarray) -> int:
        ucb = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            phi = latent[a]
            pred = float(phi @ self._linpara)
            conf = self.alpha * float(np.sqrt(max(0.0, phi @ self._A_inv @ phi)))
            ucb[a] = pred + conf
        return int(np.argmax(ucb))


class InterNeuralTS(_InterNeuralBase):
    """InterNeuralTS — 在线阶段用 LinTS（Thompson Sampling）选臂。"""

    def __init__(self, context_dim: int, n_arms: int, var: float = 0.3, **kwargs):
        super().__init__(context_dim, n_arms, **kwargs)
        self.ts_var = var

    def _select_arm(self, latent: np.ndarray) -> int:
        cov = self.ts_var * self._A_inv
        cov = 0.5 * (cov + cov.T)
        w, v = np.linalg.eigh(cov)
        w = np.maximum(w, 1e-10)
        cov_psd = (v * w) @ v.T
        try:
            linpara_s = np.random.multivariate_normal(self._linpara, cov_psd)
        except Exception:
            linpara_s = self._linpara
        vals = latent @ linpara_s   # (n_arms,)
        return int(np.argmax(vals))


# ─────────────────────────────────────────────────────────────────────────────
#  Algorithm 7: NeuralUCB — 批量运行（需预加载全部数据）
#  直接使用 InterNeuralBandit/algorithms/neural_ucb.py 中的 NeuralUCB 类
# ─────────────────────────────────────────────────────────────────────────────

def run_neural_ucb_batch(
    dataset: np.ndarray,
    context_dim: int,
    n_arms: int,
    n_rounds: int,
    opt_rewards: np.ndarray,
    hidden_size: int = 64,
    n_layers: int = 2,
    training_window: int = 100,
    train_every: int = 10,
    epochs: int = 50,
    reg_factor: float = 1.0,
    confidence_scaling_factor: float = 1.0,
    learning_rate: float = 1e-3,
) -> dict:
    """
    批量运行 external NeuralUCB（需将全部 T 步数据一次性传入）。

    数据适配：
      - features[t, a, :] = [context_t[:cap_dim]; one_hot(a, n_arms)]
        arm one-hot 编码使共享上下文数据集中不同臂得以区分
      - rewards[t, a]  = dataset[t, context_dim + a]

    返回与其他算法一致的 dict。
    """
    T = n_rounds

    # 构建 arm-augmented features
    raw_ctx = dataset[:T, :context_dim].astype(np.float32)   # (T, context_dim)
    raw_rew = dataset[:T, context_dim:context_dim + n_arms]   # (T, n_arms)

    aug_dim = context_dim + n_arms
    features = np.zeros((T, n_arms, aug_dim), dtype=np.float32)
    for a in range(n_arms):
        features[:, a, :context_dim] = raw_ctx
        features[:, a, context_dim + a] = 1.0
    rewards_mat = raw_rew.astype(np.float32)  # (T, n_arms)

    bandit = _ExtCBandit(
        T=T, n_arms=n_arms,
        n_features=aug_dim,
        features=features,
        rewards=rewards_mat,
        noise_std=1.0,
    )

    # NeuralUCB.log_output_dir 需要文件路径（非目录），写入到结果目录
    _log_file = os.path.join(RESULTS_DIR, f"_neural_ucb_log_{os.getpid()}.csv")

    neural_ucb = _ExtNeuralUCB(
        bandit=bandit,
        hidden_size=hidden_size,
        n_layers=n_layers,
        reg_factor=reg_factor,
        delta=0.1,
        confidence_scaling_factor=confidence_scaling_factor,
        training_window=training_window,
        p=0.0,
        learning_rate=learning_rate,
        epochs=epochs,
        train_every=train_every,
        throttle=100,
        log_output_dir=_log_file,
    )

    print(f"    [neural_ucb] 批量运行 {T} 步 ...", flush=True)
    t0 = time.time()
    total_regret, _timer = neural_ucb.run()
    elapsed = time.time() - t0

    # 提取 actions / rewards / regrets
    actions = neural_ucb.actions[:T].astype(np.int64)
    rewards_vec = np.array([
        rewards_mat[t, actions[t]] for t in range(T)
    ], dtype=np.float64)
    step_regret = opt_rewards[:T] - rewards_vec
    cum_regret = np.cumsum(step_regret)

    print(f"    [neural_ucb] done — cum_regret={cum_regret[-1]:.1f}  "
          f"avg_reward={np.mean(rewards_vec):.4f}  time={elapsed:.1f}s", flush=True)

    return {
        "actions": actions,
        "rewards": rewards_vec,
        "opt_rewards": opt_rewards[:T],
        "opt_actions": np.argmax(rewards_mat, axis=1),
        "step_regret": step_regret,
        "cumulative_regret": cum_regret,
        "cumulative_reward": np.cumsum(rewards_vec),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  数据集加载
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    name: str,
    n_rounds: int,
    seed: int,
    context_dim_cap: int = 100,
):
    """加载数据集，返回 (dataset, opt_rewards, opt_actions, context_dim, n_arms)。"""
    _set_seeds(seed)
    data_dir = os.path.join(ROOT, "datasets")

    total = n_rounds + 200   # 多取一些，确保不超界

    if name == "statlog":
        path = os.path.join(data_dir, "statlog.trn")
        dataset, (opt_r, opt_a) = sample_statlog_data(path, total, shuffle_rows=True)
        n_arms = 7
    elif name == "magic":
        path = os.path.join(data_dir, "magic.npz")
        dataset, (opt_r, opt_a) = sample_magic_data(total, shuffle_rows=True)
        n_arms = int(len(np.unique(opt_a)))
    elif name == "covertype":
        path = os.path.join(data_dir, "covertype.npz")
        dataset, (opt_r, opt_a) = sample_covertype_data(total, shuffle_rows=True)
        n_arms = int(len(np.unique(opt_a)))
    elif name == "mnist":
        path = os.path.join(data_dir, "mnist.npz")
        dataset, (opt_r, opt_a) = sample_mnist_data(total, shuffle_rows=True)
        n_arms = 10
    else:
        raise ValueError(f"Unknown dataset: {name}")

    actual_total = len(opt_r)
    if actual_total < n_rounds:
        warnings.warn(f"Dataset {name} only has {actual_total} rows, "
                      f"requested {n_rounds}. Truncating.")
        n_rounds = actual_total

    context_dim_raw = dataset.shape[1] - n_arms
    context_dim = min(context_dim_raw, context_dim_cap)

    # 截断上下文维度（对高维数据集如 mnist 提速）
    if context_dim < context_dim_raw:
        dataset = np.concatenate(
            [dataset[:, :context_dim], dataset[:, context_dim_raw:]], axis=1
        )
        print(f"  [{name}] context_dim truncated {context_dim_raw} → {context_dim}")

    opt_rewards = opt_r[:n_rounds]
    opt_actions = opt_a[:n_rounds]

    return dataset, opt_rewards, opt_actions, context_dim, n_arms, n_rounds

# ─────────────────────────────────────────────────────────────────────────────
#  通用在线 bandit 运行循环（LinUCB / LinTS / RP / EXP3 使用）
# ─────────────────────────────────────────────────────────────────────────────

def run_online_algo(
    algo,
    algo_name: str,
    dataset: np.ndarray,
    context_dim: int,
    n_arms: int,
    n_rounds: int,
    opt_rewards: np.ndarray,
    opt_actions: np.ndarray,
) -> dict:
    """标准在线 bandit 运行循环。

    支持两种算法接口：
      - 普通算法：algo.action(context) → arm ; algo.update(context, arm, reward)
      - InterNeural 算法：algo.step(context, rewards_all) → arm
    """
    actions_arr = np.zeros(n_rounds, dtype=np.int64)
    rewards_arr = np.zeros(n_rounds)
    t0 = time.time()
    log_freq = max(1, n_rounds // 10)

    use_step = getattr(algo, "has_step", False)

    for i in range(n_rounds):
        context = dataset[i, :context_dim].astype(np.float64)
        rewards_all = dataset[i, context_dim:context_dim + n_arms].astype(np.float64)

        if use_step:
            arm = algo.step(context, rewards_all)
        else:
            arm = algo.action(context)
            algo.update(context, arm, float(rewards_all[arm]))

        actions_arr[i] = arm
        rewards_arr[i] = float(rewards_all[arm])

        if (i + 1) % log_freq == 0:
            cum_reg = float(np.sum(opt_rewards[:i + 1]) - np.sum(rewards_arr[:i + 1]))
            print(f"    [{algo_name}] step {i + 1}/{n_rounds}  "
                  f"cum_regret={cum_reg:.1f}", flush=True)

    elapsed = time.time() - t0
    step_reg = opt_rewards - rewards_arr
    cum_reg = np.cumsum(step_reg)

    print(f"  [{algo_name}] done — cum_regret={cum_reg[-1]:.1f}  "
          f"avg_reward={np.mean(rewards_arr):.4f}  time={elapsed:.1f}s", flush=True)

    return {
        "actions": actions_arr,
        "rewards": rewards_arr,
        "opt_rewards": opt_rewards,
        "opt_actions": opt_actions,
        "step_regret": step_reg,
        "cumulative_regret": cum_reg,
        "cumulative_reward": np.cumsum(rewards_arr),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  算法工厂：根据数据集参数创建算法实例
# ─────────────────────────────────────────────────────────────────────────────

def build_algo(
    name: str,
    context_dim: int,
    n_arms: int,
    latent_dim: int = 10,
    alpha: float = 0.5,
    ts_var: float = 0.3,
    rp_dim: int = 10,
    offline_lr: float = 1e-3,
    offline_epochs: int = 50,
    online_max_step: int = 50,
    online_batch_size: int = 50,
    historical_data_size: int = 100,
    exp3_gamma: float = 0.1,
):
    if name == "lin_ucb":
        return LinUCB(context_dim, n_arms, alpha=alpha)
    elif name == "lin_ts":
        return LinTS(context_dim, n_arms, var=ts_var)
    elif name == "lin_ucb_rp":
        return LinUCBRP(context_dim, n_arms, rp_dim=max(rp_dim, n_arms))
    elif name == "exp3":
        return Exp3(n_arms, gamma=exp3_gamma)
    elif name == "inter_neural_ucb":
        return InterNeuralUCB(
            context_dim, n_arms, alpha=alpha, latent_dim=latent_dim,
            offline_lr=offline_lr, offline_epochs=offline_epochs,
            online_max_step=online_max_step, online_batch_size=online_batch_size,
            historical_data_size=historical_data_size,
        )
    elif name == "inter_neural_ts":
        return InterNeuralTS(
            context_dim, n_arms, var=ts_var, latent_dim=latent_dim,
            offline_lr=offline_lr, offline_epochs=offline_epochs,
            online_max_step=online_max_step, online_batch_size=online_batch_size,
            historical_data_size=historical_data_size,
        )
    else:
        raise ValueError(f"Unknown algorithm: {name}")

# ─────────────────────────────────────────────────────────────────────────────
#  保存 / 加载结果
# ─────────────────────────────────────────────────────────────────────────────

def _result_dir(dataset_name: str, n_rounds: int, seed: int) -> str:
    d = os.path.join(RESULTS_DIR, f"{dataset_name}_{n_rounds}r_seed{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def save_result(dataset_name: str, algo_name: str, data: dict, n_rounds: int, seed: int):
    exp_d = _result_dir(dataset_name, n_rounds, seed)
    fp = os.path.join(exp_d, f"{algo_name}.npz")
    np.savez_compressed(fp, **{k: v for k, v in data.items()})
    print(f"  Saved: {fp}")
    return fp


def load_result(dataset_name: str, algo_name: str, n_rounds: int, seed: int) -> dict | None:
    fp = os.path.join(_result_dir(dataset_name, n_rounds, seed), f"{algo_name}.npz")
    if not os.path.exists(fp):
        return None
    d = np.load(fp, allow_pickle=True)
    return {k: d[k] for k in d.files}

# ─────────────────────────────────────────────────────────────────────────────
#  绘图
# ─────────────────────────────────────────────────────────────────────────────

_STYLES = {
    "lin_ucb":           {"color": "#d62728", "ls": "--",  "lw": 1.5},
    "lin_ts":            {"color": "#1f77b4", "ls": "--",  "lw": 1.5},
    "lin_ucb_rp":        {"color": "#9467bd", "ls": "--",  "lw": 1.5},
    "exp3":              {"color": "#2ca02c", "ls": ":",   "lw": 1.5},
    "inter_neural_ucb":  {"color": "#e377c2", "ls": "-",   "lw": 2.2},
    "inter_neural_ts":   {"color": "#ff7f0e", "ls": "-",   "lw": 2.2},
    "neural_ucb":        {"color": "#17becf", "ls": "-.",  "lw": 2.0},
}

_LABELS = {
    "lin_ucb":           "LinUCB",
    "lin_ts":            "LinTS",
    "lin_ucb_rp":        "LinUCB+RP",
    "exp3":              "EXP3",
    "inter_neural_ucb":  "InterNeural-UCB",
    "inter_neural_ts":   "InterNeural-TS",
    "neural_ucb":        "NeuralUCB",
}


def plot_results(
    all_data: dict[str, dict],
    dataset_name: str,
    n_rounds: int,
    seed: int,
    out_dir: str,
):
    """生成累计 Regret 和累计 Reward 两张对比图。"""
    steps = np.arange(1, n_rounds + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for algo_name, data in all_data.items():
        style = _STYLES.get(algo_name, {"color": "gray", "ls": "-", "lw": 1.5})
        label = _LABELS.get(algo_name, algo_name)
        n = len(data["cumulative_regret"])
        s = steps[:n]

        axes[0].plot(s, data["cumulative_regret"][:n], label=label, **style)
        axes[1].plot(s, data["cumulative_reward"][:n],  label=label, **style)

    for ax, ylabel, title_suffix in zip(
        axes,
        ["Cumulative Regret", "Cumulative Reward"],
        ["Cumulative Regret", "Cumulative Reward"],
    ):
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(
            f"{dataset_name.upper()} — {title_suffix} ({n_rounds}r, seed={seed})",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fp = os.path.join(out_dir, f"comparison_{n_rounds}r_seed{seed}.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {fp}")
    return fp


def print_summary(all_data: dict[str, dict], n_rounds: int):
    print(f"\n{'=' * 72}")
    print(f"{'Algorithm':<25s} {'Cum.Regret':>13s} {'Avg.Reward':>13s} {'Accuracy':>10s}")
    print(f"{'=' * 72}")
    for name, data in all_data.items():
        cr = float(data["cumulative_regret"][-1])
        avg_r = float(np.mean(data["rewards"]))
        n = len(data["actions"])
        acc = float(np.mean(data["actions"][:n] == data["opt_actions"][:n]))
        label = _LABELS.get(name, name)
        print(f"  {label:<23s} {cr:>13.1f} {avg_r:>13.4f} {acc:>9.3f}")
    print(f"{'=' * 72}\n")

# ─────────────────────────────────────────────────────────────────────────────
#  主流程：对单个数据集运行所有算法
# ─────────────────────────────────────────────────────────────────────────────

ONLINE_ALGOS = [
    "lin_ucb",
    "lin_ts",
    "lin_ucb_rp",
    "exp3",
    "inter_neural_ucb",
    "inter_neural_ts",
]
ALL_ALGOS = ONLINE_ALGOS + ["neural_ucb"]


def run_dataset(
    dataset_name: str,
    n_rounds: int,
    seed: int,
    skip: list[str] | None = None,
    force: bool = False,
    context_dim_cap: int = 100,
    # 超参数
    latent_dim: int = 10,
    alpha: float = 0.5,
    ts_var: float = 0.3,
    offline_epochs: int = 50,
    online_max_step: int = 50,
    online_batch_size: int = 50,
    historical_data_size: int = 100,
    offline_lr: float = 1e-3,
    # neural_ucb 专属
    neural_ucb_hidden: int = 64,
    neural_ucb_layers: int = 2,
    neural_ucb_epochs: int = 50,
    neural_ucb_train_every: int = 10,
):
    skip = set(skip or [])
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}  n_rounds={n_rounds}  seed={seed}")
    print(f"{'='*60}")

    # 加载数据
    dataset, opt_rewards, opt_actions, context_dim, n_arms, n_rounds = load_dataset(
        dataset_name, n_rounds, seed, context_dim_cap=context_dim_cap
    )
    print(f"  context_dim={context_dim}, n_arms={n_arms}, n_rounds={n_rounds}")

    exp_d = _result_dir(dataset_name, n_rounds, seed)
    all_data: dict[str, dict] = {}

    # ── 在线算法 ──────────────────────────────────────────────────────────────
    for algo_name in ONLINE_ALGOS:
        if algo_name in skip:
            print(f"  [skip] {algo_name}")
            continue

        cached = load_result(dataset_name, algo_name, n_rounds, seed)
        if cached is not None and not force:
            print(f"  [cached] {algo_name}")
            all_data[algo_name] = cached
            continue

        print(f"\n  Running {_LABELS.get(algo_name, algo_name)} ...")
        _set_seeds(seed)

        algo = build_algo(
            algo_name,
            context_dim, n_arms,
            latent_dim=latent_dim,
            alpha=alpha, ts_var=ts_var,
            offline_lr=offline_lr,
            offline_epochs=offline_epochs,
            online_max_step=online_max_step,
            online_batch_size=online_batch_size,
            historical_data_size=historical_data_size,
        )

        result = run_online_algo(
            algo, algo_name,
            dataset, context_dim, n_arms, n_rounds,
            opt_rewards, opt_actions,
        )
        save_result(dataset_name, algo_name, result, n_rounds, seed)
        all_data[algo_name] = result

    # ── NeuralUCB（批量模式）──────────────────────────────────────────────────
    if "neural_ucb" not in skip:
        cached = load_result(dataset_name, "neural_ucb", n_rounds, seed)
        if cached is not None and not force:
            print(f"  [cached] neural_ucb")
            all_data["neural_ucb"] = cached
        else:
            print(f"\n  Running NeuralUCB (batch) ...")
            _set_seeds(seed)
            result = run_neural_ucb_batch(
                dataset=dataset,
                context_dim=context_dim,
                n_arms=n_arms,
                n_rounds=n_rounds,
                opt_rewards=opt_rewards,
                hidden_size=neural_ucb_hidden,
                n_layers=neural_ucb_layers,
                epochs=neural_ucb_epochs,
                train_every=neural_ucb_train_every,
            )
            save_result(dataset_name, "neural_ucb", result, n_rounds, seed)
            all_data["neural_ucb"] = result
    else:
        print(f"  [skip] neural_ucb")

    # ── 打印摘要并绘图 ─────────────────────────────────────────────────────────
    if all_data:
        print_summary(all_data, n_rounds)
        plot_results(all_data, dataset_name, n_rounds, seed, exp_d)

    return all_data

# ─────────────────────────────────────────────────────────────────────────────
#  命令行入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run InterNeuralBandit algorithms on our datasets"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["statlog", "magic", "covertype", "mnist"],
        help="Datasets to run: statlog magic covertype mnist",
    )
    parser.add_argument("--n_rounds", type=int, default=1000,
                        help="Number of bandit rounds per dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Ignore cached results and rerun everything")
    parser.add_argument("--plot_only", action="store_true",
                        help="Only regenerate plots from cached .npz files")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=ALL_ALGOS,
        help="Algorithms to skip",
    )
    parser.add_argument("--context_dim_cap", type=int, default=100,
                        help="Max context dimension (truncate larger datasets, e.g. mnist)")

    # NN 超参数
    parser.add_argument("--latent_dim", type=int, default=10,
                        help="Latent dimension for InterNeural NN")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="UCB exploration coefficient")
    parser.add_argument("--ts_var", type=float, default=0.3,
                        help="Thompson Sampling posterior variance scaling")
    parser.add_argument("--offline_epochs", type=int, default=50,
                        help="NN offline training epochs per iteration")
    parser.add_argument("--online_max_step", type=int, default=50,
                        help="Online steps per InterNeural iteration")
    parser.add_argument("--online_batch_size", type=int, default=50,
                        help="History warmup steps per InterNeural iteration")
    parser.add_argument("--offline_lr", type=float, default=1e-3)

    # neural_ucb 专属
    parser.add_argument("--neural_ucb_hidden", type=int, default=64)
    parser.add_argument("--neural_ucb_layers", type=int, default=2)
    parser.add_argument("--neural_ucb_epochs", type=int, default=50)
    parser.add_argument("--neural_ucb_train_every", type=int, default=10)

    args = parser.parse_args()

    if args.plot_only:
        # 仅重绘图，复用缓存
        for dname in args.datasets:
            exp_d = _result_dir(dname, args.n_rounds, args.seed)
            all_data = {}
            for algo_name in ALL_ALGOS:
                cached = load_result(dname, algo_name, args.n_rounds, args.seed)
                if cached is not None:
                    all_data[algo_name] = cached
            if all_data:
                print(f"\n[{dname}] Plotting {list(all_data.keys())} ...")
                print_summary(all_data, args.n_rounds)
                plot_results(all_data, dname, args.n_rounds, args.seed, exp_d)
            else:
                print(f"[{dname}] No cached results found. Run without --plot_only first.")
        return

    for dname in args.datasets:
        run_dataset(
            dataset_name=dname.lower(),
            n_rounds=args.n_rounds,
            seed=args.seed,
            skip=args.skip,
            force=args.force,
            context_dim_cap=args.context_dim_cap,
            latent_dim=args.latent_dim,
            alpha=args.alpha,
            ts_var=args.ts_var,
            offline_epochs=args.offline_epochs,
            online_max_step=args.online_max_step,
            online_batch_size=args.online_batch_size,
            historical_data_size=args.online_batch_size * 2,
            offline_lr=args.offline_lr,
            neural_ucb_hidden=args.neural_ucb_hidden,
            neural_ucb_layers=args.neural_ucb_layers,
            neural_ucb_epochs=args.neural_ucb_epochs,
            neural_ucb_train_every=args.neural_ucb_train_every,
        )


if __name__ == "__main__":
    main()
