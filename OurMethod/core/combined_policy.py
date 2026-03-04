"""Step 3 — Combined UCB 策略：三个完整 UCB 取 min。

核心公式：
  A_t ∈ argmax_{a∈A} { min{ UCB_t^{s1}(a), UCB_t^{s2}(a), UCB_t^{s3}(a) } }

其中每个 UCB 都是**完整**的 exploit + explore：
  UCB^s1(a) = θ_a^T z_t + α * sqrt(z_t^T A_a^{-1} z_t)              (Neural LinUCB)
  UCB^s2(a) = μ̂_a      + c1 * sqrt(ln(t)/N_a)                       (经典 UCB1)
  UCB^s3(a) = μ^{on+off}(a)
            + c_sim * sqrt(2*log(2t/δ_t) / (N(a)+T_s(a)))
            + T_s(a)/(N(a)+T_s(a)) * V(a)                          (LLM 仿真)

 UCB^s3 各项解释：
   μ^{on+off}(a) = (sum_online+sum_llm) / (N+T_s)     合并在线+离线均值
   sqrt(2log(2t/δ_t)/(N+T_s))                        Hoeffding 置信半径
   T_s/(N+T_s)*V(a)                                   加权偏差修正
   V(a) = |mu_llm(a) - mu_online(a)|                  LLM 与在线均值差异

设计动机：
  冷启动 → N=0, UCB^s3 = mu_llm + confidence, LLM 先验主导
  数据增多 → V(a) 度量偏差，但权重 T_s/(N+T_s) 衰减
  最终 → N>>T_s, UCB^s3 收敛至标准 Hoeffding UCB
  Neural LinUCB 收敛最快 → UCB^s1 收紧 → 逐渐主导
"""

from __future__ import annotations

import math
import sys, os
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

from typing import Dict, List, Optional, Tuple

import numpy as np

from OurMethod.core.protocol import (
    Arm, Context, DecisionRecord, DsLlm, Policy,
)
from OurMethod.core.sim_stats import SimulationStats


class CombinedUCBPolicy(Policy):
    """三个完整 UCB 取 min 的 Combined UCB 策略。

    A_t = argmax_a { min(UCB^s1(a), UCB^s2(a), UCB^s3(a)) }
    """

    def __init__(
        self,
        num_actions: int,
        z_dim: int,
        c_ucb1: float = 1.0,
        alpha_linucb: float = 1.0,
        c_sim: float = 1.0,
        lambda_reg: float = 1.0,
        initial_pulls: int = 1,
        name: str = "combined_ucb",
    ):
        # ============================================================ #
        #  参数指南 (Parameter Guide)
        # ============================================================ #
        #
        # 🔴 重点关注 — 直接影响探索/利用权衡与最终性能
        #
        #   c_ucb1 (float, default=1.0)
        #       UCB1 探索系数，控制 UCB^s2 的探索半径。
        #       增大 → 更多探索（适合高噪声环境）；减小 → 更早收敛。
        #       典型范围: [0.1, 5.0]，建议从 1.0 开始调。
        #
        #   alpha_linucb (float, default=1.0)
        #       Neural LinUCB 探索系数，控制 UCB^s1 的置信半径。
        #       增大 → 对特征不确定性更敏感；减小 → 更信任当前 θ 估计。
        #       典型范围: [0.1, 3.0]，特征维度高时建议稍小。
        #
        #   c_sim (float, default=1.0)
        #       LLM 仿真 UCB 探索系数，控制 UCB^s3 的探索半径。
        #       增大 → 冷启动时 LLM 约束更宽松，允许更多探索；
        #       减小 → 更信任 LLM 预测，冷启动约束更紧。
        #       典型范围: [0.5, 3.0]，LLM 质量高时可减小。
        #
        #   z_dim (int)
        #       压缩特征维度，即 z_t 的长度。
        #       太大 → LinUCB 收敛慢，A_a^{-1} 计算量大；
        #       太小 → 信息丢失，θ 无法精确拟合。
        #       典型范围: [8, 64]，建议 16~32 作为起点。
        #
        # 🟡 可按需调整
        #
        #   lambda_reg (float, default=1.0)
        #       LinUCB 正则化系数，初始化 A_a = λI。
        #       增大 → θ 估计更保守（防过拟合）；减小 → 初期收敛更快。
        #       通常保持 1.0，除非特征尺度异常。
        #
        #   initial_pulls (int, default=1)
        #       冷启动 round-robin 轮数：前 K*initial_pulls 步每个 arm 轮流拉。
        #       确保所有 arm 至少被观测一次。1 通常足够。
        #
        # 🟢 无需调整
        #
        #   num_actions (int)  — 由数据集/环境决定
        #   name (str)         — 仅用于日志标识
        # ============================================================ #
        self.name = name
        self.K = num_actions
        self.z_dim = z_dim
        self.c_ucb1 = c_ucb1
        self.alpha_linucb = alpha_linucb
        self.lambda_reg = lambda_reg
        self.initial_pulls = initial_pulls

        # ---- UCB1 统计 ---- #
        self.counts = np.zeros(num_actions, dtype=np.float64)
        self.sum_rewards = np.zeros(num_actions, dtype=np.float64)

        # ---- Neural LinUCB 统计 ---- #
        self.A: List[np.ndarray] = [
            lambda_reg * np.eye(z_dim) for _ in range(num_actions)
        ]
        self.b_vec: List[np.ndarray] = [
            np.zeros(z_dim) for _ in range(num_actions)
        ]
        self.A_inv: List[np.ndarray] = [
            (1.0 / lambda_reg) * np.eye(z_dim) for _ in range(num_actions)
        ]

        # ---- LLM 仿真统计 ---- #
        self.sim_stats = SimulationStats(num_actions, c_sim=c_sim)

        self.t: int = 0

    # ================================================================== #
    #  核心接口                                                           #
    # ================================================================== #
    def select(
        self,
        context: Context,
        arms: List[Arm],
        z_t: np.ndarray,
        ds_llm: Optional[List[DsLlm]] = None,
    ) -> int:
        # ---- 如需要 ds_llm 更新仿真统计 ---- #
        if ds_llm:
            self.sim_stats.update_from_ds_llm(ds_llm)

        # ---- 冷启动：round-robin ---- #
        if self.t < self.K * self.initial_pulls:
            return self.t % self.K

        scores = np.zeros(self.K)
        ucb_s1_list: List[float] = []   # Neural LinUCB 完整 UCB
        ucb_s2_list: List[float] = []   # UCB1 完整 UCB
        ucb_s3_list: List[float] = []   # LLM 仿真完整 UCB

        for a in range(self.K):
            # ---- UCB^s1: Neural LinUCB ---- #
            #   θ_a^T z_t + α * sqrt(z_t^T A_a^{-1} z_t)
            theta_a = self.A_inv[a] @ self.b_vec[a]
            exploit_linucb = float(z_t @ theta_a)
            radius_linucb = self._radius_linucb(a, z_t)
            ucb_s1 = exploit_linucb + radius_linucb

            # ---- UCB^s2: UCB1 ---- #
            #   μ̂_a + c1 * sqrt(ln(t)/N_a)
            mu_hat = (self.sum_rewards[a] / self.counts[a]
                      if self.counts[a] > 0 else 0.0)
            radius_ucb1 = self._radius_ucb1(a)
            ucb_s2 = mu_hat + radius_ucb1

            # ---- UCB^s3: LLM 仿真 UCB ---- #
            #   μ^{on+off}(a) + c_sim*sqrt(2log(2t/δ_t)/(N+T_s)) + T_s/(N+T_s)*V(a)
            ucb_s3 = self.sim_stats.ucb_llm(a)

            ucb_s1_list.append(ucb_s1)
            ucb_s2_list.append(ucb_s2)
            ucb_s3_list.append(ucb_s3)

            # ---- 取三个完整 UCB 的 min ---- #
            scores[a] = min(ucb_s1, ucb_s2, ucb_s3)

        # 缓存调试信息供 runner / 日志读取
        self._last_ucb_values = scores.tolist()
        self._last_ucb_s1 = ucb_s1_list
        self._last_ucb_s2 = ucb_s2_list
        self._last_ucb_s3 = ucb_s3_list
        self._last_r_ucb1 = [self._radius_ucb1(a) for a in range(self.K)]
        self._last_r_linucb = [self._radius_linucb(a, z_t) for a in range(self.K)]
        self._last_r_llm = [self.sim_stats.radius_llm(a) for a in range(self.K)]

        return int(np.argmax(scores))

    def update(self, record: DecisionRecord):
        a = record.chosen_arm
        r = record.reward
        z = record.z_t
        self.t += 1

        # ---- UCB1 ---- #
        self.counts[a] += 1
        self.sum_rewards[a] += r

        # ---- LinUCB ---- #
        if z is not None:
            self.A[a] += np.outer(z, z)
            self.b_vec[a] += r * z
            # Sherman-Morrison
            Az = self.A_inv[a] @ z
            self.A_inv[a] -= np.outer(Az, Az) / (1.0 + z @ Az)

        # ---- 仿真统计 ---- #
        self.sim_stats.update_online(a, r)

    # ================================================================== #
    #  半径计算                                                           #
    # ================================================================== #
    def _radius_ucb1(self, arm: int) -> float:
        if self.counts[arm] == 0:
            return float("inf")
        return self.c_ucb1 * math.sqrt(math.log(self.t + 1) / self.counts[arm])

    def _radius_linucb(self, arm: int, z: np.ndarray) -> float:
        return self.alpha_linucb * math.sqrt(float(z @ self.A_inv[arm] @ z))

    # ================================================================== #
    #  工具方法                                                           #
    # ================================================================== #
    def get_last_debug(self) -> Dict[str, Optional[List[float]]]:
        return {
            "ucb_values": getattr(self, "_last_ucb_values", None),
            "ucb_s1": getattr(self, "_last_ucb_s1", None),        # Neural LinUCB 完整 UCB
            "ucb_s2": getattr(self, "_last_ucb_s2", None),        # UCB1 完整 UCB
            "ucb_s3": getattr(self, "_last_ucb_s3", None),        # LLM 仿真完整 UCB
            "radius_ucb1": getattr(self, "_last_r_ucb1", None),
            "radius_linucb": getattr(self, "_last_r_linucb", None),
            "radius_llm": getattr(self, "_last_r_llm", None),
        }

    def get_theta(self, arm: int) -> np.ndarray:
        return self.A_inv[arm] @ self.b_vec[arm]

    def rebuild_linucb(
        self,
        all_z: np.ndarray,
        all_arms: np.ndarray,
        all_rewards: np.ndarray,
    ):
        """用新 z 重建 LinUCB 统计量（离线训练 fφ 后调用）。"""
        d = self.z_dim
        self.A = [self.lambda_reg * np.eye(d) for _ in range(self.K)]
        self.b_vec = [np.zeros(d) for _ in range(self.K)]
        for i in range(len(all_z)):
            a = int(all_arms[i])
            r = float(all_rewards[i])
            z = all_z[i]
            self.A[a] += np.outer(z, z)
            self.b_vec[a] += r * z
        self.A_inv = [np.linalg.inv(self.A[a]) for a in range(self.K)]

    def reset(self):
        d = self.z_dim
        self.counts[:] = 0
        self.sum_rewards[:] = 0
        self.A = [self.lambda_reg * np.eye(d) for _ in range(self.K)]
        self.b_vec = [np.zeros(d) for _ in range(self.K)]
        self.A_inv = [(1.0 / self.lambda_reg) * np.eye(d) for _ in range(self.K)]
        self.t = 0
