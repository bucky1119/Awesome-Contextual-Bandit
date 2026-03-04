"""Step 3 — 仿真统计跟踪 + LLM 仿真 UCB (修订版)。

维护每个 arm 的：
  N_online(a)   在线拉取次数
  sum_online(a)  在线奖励累计
  T_s(a)        LLM 离线样本次数 (冷启动生成 + 在线生成)
  sum_llm(a)    LLM 离线预测均值累计

修订后的 UCB^s3 公式：
  UCB^s3_t(a) = μ^{on+off}(a)
              + c_sim * sqrt( 2*log(2t/δ_t) / (N_t(a) + T_s(a)) )
              + T_s(a) / (N_t(a) + T_s(a)) * V(a)

其中：
  μ^{on+off}(a) = (sum_online + sum_llm) / (N + T_s)   合并均值 (在线+离线)
  δ_t = 1/t                                            置信参数 (Hoeffding)
  V(a) = |mu_llm(a) - mu_online(a)|                    LLM-在线偏差
      = 0 当 N_online=0 或 T_s=0 (无法度量偏差)
  T_s/(N+T_s)                                           衰减权重：在线数据越多，偏差修正越小

设计动机：
  冷启动  → N=0, UCB^s3 = mu_llm + confidence, LLM 先验主导
  数据增多 → V(a) 偏差被度量并修正，且权重衰减
  长期    → N>>T_s, 偏差项→0, 合并均值→在线均值, UCB^s3→标准 Hoeffding UCB
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

import sys, os
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

from OurMethod.core.protocol import DsLlm


class SimulationStats:
    """跟踪在线/离线统计并计算 LLM 仿真 UCB（修订公式）。"""

    def __init__(self, num_actions: int, c_sim: float = 1.0):
        # ============================================================ #
        #  参数指南:
        #  c_sim (float, default=1.0) — UCB^s3 Hoeffding 置信半径乘子。
        #      由 CombinedUCBPolicy 传入，见其参数指南。
        #  num_actions — 由数据集决定，无需手动调整。
        # ============================================================ #
        self.K = num_actions
        self.c_sim = c_sim

        # 在线统计
        self.sum_online = np.zeros(num_actions, dtype=np.float64)
        self.n_online = np.zeros(num_actions, dtype=np.float64)

        # LLM 离线统计（增量均值）
        self.sum_llm = np.zeros(num_actions, dtype=np.float64)
        self.t_s = np.zeros(num_actions, dtype=np.float64)       # T_s(a)

        self.total_t: int = 0                                     # 全局时间步

    # ---- 在线更新 ---- #
    def update_online(self, arm: int, reward: float):
        self.n_online[arm] += 1
        self.sum_online[arm] += reward
        self.total_t += 1

    # ---- LLM 离线更新 (来自 ds_llm) ---- #
    def update_from_ds_llm(self, ds_llm: List[DsLlm]):
        for entry in ds_llm:
            a = entry.arm_id
            if 0 <= a < self.K:
                self.t_s[a] += 1
                self.sum_llm[a] += entry.predicted_mean

    # ---- 查询 ---- #
    def mu_online(self, arm: int) -> float:
        if self.n_online[arm] == 0:
            return 0.0
        return float(self.sum_online[arm] / self.n_online[arm])

    def mu_llm(self, arm: int) -> float:
        if self.t_s[arm] == 0:
            return 0.0
        return float(self.sum_llm[arm] / self.t_s[arm])

    def mu_combined(self, arm: int) -> float:
        """μ^{on+off}(a) = (sum_online + sum_llm) / (N_online + T_s)。

        合并在线+离线样本的均值。N+T_s=0 时返回 0.0。
        """
        total_n = self.n_online[arm] + self.t_s[arm]
        if total_n == 0:
            return 0.0
        return float((self.sum_online[arm] + self.sum_llm[arm]) / total_n)

    def V(self, arm: int) -> float:
        """V(a) = |mu_llm(a) - mu_online(a)|.

        度量 LLM 预测与真实在线奖励的偏差。
        当 N_online=0 或 T_s=0 时无法度量，返回 0。
        """
        if self.n_online[arm] == 0 or self.t_s[arm] == 0:
            return 0.0
        return abs(self.mu_llm(arm) - self.mu_online(arm))

    def radius_llm(self, arm: int) -> float:
        """UCB^s3 的半径部分 (不含 μ^{on+off})。

        = c_sim * sqrt(2*log(2t/δ_t) / (N+T_s)) + T_s/(N+T_s) * V(a)
        其中 δ_t = 1/t。N+T_s=0 时返回 +inf。
        """
        N_a = self.n_online[arm]
        T_s_a = self.t_s[arm]
        total_n = N_a + T_s_a
        if total_n == 0:
            return float("inf")
        t = max(self.total_t, 1)
        # δ_t = 1/t  →  2t/δ_t = 2t²
        log_term = 2.0 * math.log(2.0 * t * t)
        confidence = self.c_sim * math.sqrt(max(log_term, 0.0) / total_n)
        # 加权偏差修正
        bias_weight = T_s_a / total_n
        bias = bias_weight * self.V(arm)
        return confidence + bias

    def ucb_llm(self, arm: int) -> float:
        """完整 UCB^s3:

        UCB^s3(a) = μ^{on+off}(a)
                  + c_sim * sqrt(2*log(2t/δ_t) / (N+T_s))
                  + T_s/(N+T_s) * V(a)

        N+T_s=0 时返回 +inf（无约束）。
        """
        N_a = self.n_online[arm]
        T_s_a = self.t_s[arm]
        total_n = N_a + T_s_a
        if total_n == 0:
            return float("inf")
        return self.mu_combined(arm) + self.radius_llm(arm)

    def all_radii_llm(self) -> List[float]:
        return [self.radius_llm(a) for a in range(self.K)]

    def summary(self) -> Dict[str, object]:
        return {
            "total_t": self.total_t,
            "n_online": self.n_online.tolist(),
            "t_s": self.t_s.tolist(),
            "V": [self.V(a) for a in range(self.K)],
            "mu_combined": [self.mu_combined(a) for a in range(self.K)],
        }
