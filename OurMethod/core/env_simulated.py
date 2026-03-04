"""环境仿真模块：给定 arm 隐藏权重生成 reward。

提供两种仿真模式：
1. 线性仿真：reward = context^T w_a + noise
2. 非线性仿真：reward = σ(context^T w_a) + noise

用于：
- 离线实验验证算法正确性
- LLM 仿真数据生成的 ground-truth 参考
- 冷启动阶段的仿真数据生产
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np

from OurMethod.core.protocol import Arm


class SimulatedEnvironment:
    """基于隐藏权重的仿真环境。

    每个 arm 有一个隐藏权重向量 w_a ∈ R^d，
    奖励生成方式为 reward = f(context^T w_a) + ε。
    """

    def __init__(
        self,
        num_actions: int,
        context_dim: int,
        noise_std: float = 0.1,
        reward_type: str = "linear",
        seed: Optional[int] = None,
    ):
        """初始化仿真环境。

        Args:
            num_actions: arm 数量
            context_dim: 上下文维度
            noise_std: 高斯噪声标准差
            reward_type: "linear" 或 "sigmoid"
            seed: 随机种子
        """
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.noise_std = noise_std
        self.reward_type = reward_type
        self.rng = np.random.RandomState(seed)

        # 随机生成各 arm 的隐藏权重
        self.weights: List[np.ndarray] = [
            self.rng.randn(context_dim).astype(np.float64) * 0.5
            for _ in range(num_actions)
        ]

        # 创建 Arm 对象
        self.arms: List[Arm] = [
            Arm(
                arm_id=i,
                name=f"arm_{i}",
                description=f"Action {i} with hidden linear weight",
            )
            for i in range(num_actions)
        ]

    # ------------------------------------------------------------------ #
    #                        奖励生成                                      #
    # ------------------------------------------------------------------ #

    def reward(self, context: np.ndarray, action: int) -> float:
        """为给定 (context, action) 生成奖励。

        Args:
            context: 上下文特征向量，shape=(d,)
            action: arm ID

        Returns:
            浮点数奖励
        """
        raw = context @ self.weights[action]
        if self.reward_type == "sigmoid":
            raw = 1.0 / (1.0 + np.exp(-raw))
        noise = self.rng.randn() * self.noise_std
        return float(raw + noise)

    def optimal_reward(self, context: np.ndarray) -> Tuple[float, int]:
        """计算给定上下文下的最优奖励和最优 arm。

        Args:
            context: 上下文特征向量

        Returns:
            (最优奖励（无噪声）, 最优 arm ID)
        """
        expected = np.array([context @ w for w in self.weights])
        if self.reward_type == "sigmoid":
            expected = 1.0 / (1.0 + np.exp(-expected))
        best_arm = int(np.argmax(expected))
        return float(expected[best_arm]), best_arm

    # ------------------------------------------------------------------ #
    #                        批量数据生成                                  #
    # ------------------------------------------------------------------ #

    def generate_dataset(
        self, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """生成批量仿真数据集。

        Args:
            n_samples: 样本数

        Returns:
            (contexts, rewards_matrix, opt_rewards, opt_actions)
            - contexts: (n, d)
            - rewards_matrix: (n, K)，每个 arm 的奖励（含噪声）
            - opt_rewards: (n,) 最优奖励
            - opt_actions: (n,) 最优动作
        """
        contexts = self.rng.randn(n_samples, self.context_dim).astype(np.float64)
        rewards_matrix = np.zeros((n_samples, self.num_actions))
        opt_rewards = np.zeros(n_samples)
        opt_actions = np.zeros(n_samples, dtype=np.int64)

        for i in range(n_samples):
            for a in range(self.num_actions):
                rewards_matrix[i, a] = self.reward(contexts[i], a)
            opt_r, opt_a = self.optimal_reward(contexts[i])
            opt_rewards[i] = opt_r
            opt_actions[i] = opt_a

        return contexts, rewards_matrix, opt_rewards, opt_actions

    def generate_simulated_interactions(
        self, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成随机交互数据（用于冷启动 warm_start）。

        随机选择 arm，记录奖励。

        Returns:
            (contexts, actions, rewards) — 每个 shape=(n,) 或 (n, d)
        """
        contexts = self.rng.randn(n_samples, self.context_dim).astype(np.float64)
        actions = self.rng.randint(0, self.num_actions, size=n_samples)
        rewards = np.array([
            self.reward(contexts[i], actions[i]) for i in range(n_samples)
        ])
        return contexts, actions, rewards

    # ------------------------------------------------------------------ #
    #                        兼容接口                                      #
    # ------------------------------------------------------------------ #

    def to_bandit_dataset(self, n_samples: int) -> np.ndarray:
        """生成与现有项目 run_contextual_bandit 兼容的数据集。

        Returns:
            dataset: shape=(n, d+K)，前 d 列为上下文，后 K 列为各 arm 的奖励
        """
        contexts, rewards_matrix, _, _ = self.generate_dataset(n_samples)
        return np.hstack([contexts, rewards_matrix])
