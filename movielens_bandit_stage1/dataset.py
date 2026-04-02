"""Dataset wrapper for paper-style MovieLens contextual bandit (fixed arms)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class MovieLensBanditDataset:
    """
    Paper-style contextual bandit dataset.

    Each round:
        context = current movie genre vector
        arms    = fixed users
        reward  = Bernoulli(mu[t, i])

    Data layout:
        arm_features: [N, d_arm]
        contexts:     [T, d_ctx]
        mu_matrix:    [T, N]
    """

    arm_features: np.ndarray        # [N, d_arm]
    contexts: np.ndarray            # [T, d_ctx]
    mu_matrix: np.ndarray           # [T, N]

    selected_user_ids: np.ndarray   # [N]
    movie_ids: np.ndarray           # [T]

    num_arms: int
    context_dim: int
    arm_dim: int

    feature_schema: Dict[str, Any]

    # =========================
    # loading
    # =========================
    @classmethod
    def from_npz(cls, npz_path: str) -> "MovieLensBanditDataset":
        raw = np.load(npz_path, allow_pickle=True)

        schema_raw = raw["feature_schema_json"]
        if isinstance(schema_raw, np.ndarray):
            schema_text = str(schema_raw.item())
        else:
            schema_text = str(schema_raw)

        arm_features = raw["arm_features"].astype(np.float32)
        contexts = raw["contexts"].astype(np.float32)
        mu_matrix = raw["mu_matrix"].astype(np.float32)

        selected_user_ids = raw["selected_user_ids"].astype(np.int64)
        movie_ids = raw["movie_ids"].astype(np.int64)

        num_arms = int(raw["num_arms"].item())
        context_dim = int(raw["genre_dim"].item())
        arm_dim = arm_features.shape[1]

        return cls(
            arm_features=arm_features,
            contexts=contexts,
            mu_matrix=mu_matrix,
            selected_user_ids=selected_user_ids,
            movie_ids=movie_ids,
            num_arms=num_arms,
            context_dim=context_dim,
            arm_dim=arm_dim,
            feature_schema=json.loads(schema_text),
        )

    # =========================
    # basic
    # =========================
    def __len__(self) -> int:
        return int(self.contexts.shape[0])

    # =========================
    # round access
    # =========================
    def get_round(self, t: int) -> Dict[str, Any]:
        if t < 0 or t >= len(self):
            raise IndexError(f"Round index out of range: {t}")

        context = self.contexts[t]         # [d_ctx]
        mu = self.mu_matrix[t]             # [N]

        return {
            "round_id": t,
            "movie_id": int(self.movie_ids[t]),
            "context": context.copy(),            # 当前电影特征
            "arm_features": self.arm_features.copy(),  # 所有用户特征
            "mu": mu.copy(),                      # 所有 arm 的真实期望奖励
            "meta": {
                "num_arms": self.num_arms,
                "context_dim": self.context_dim,
                "arm_dim": self.arm_dim,
            },
        }

    # =========================
    # utility
    # =========================
    def sample_reward(self, t: int, arm: int, rng: np.random.Generator | None = None) -> float:
        """
        Sample Bernoulli reward for selected arm.
        """
        if rng is None:
            rng = np.random.default_rng()

        mu = self.mu_matrix[t, arm]
        return float(rng.random() < mu)

    def optimal_arm(self, t: int) -> int:
        """
        Return optimal arm index (argmax mu).
        """
        return int(np.argmax(self.mu_matrix[t]))

    def regret(self, t: int, chosen_arm: int) -> float:
        """
        Pseudo-regret using expected rewards (mu).
        """
        mu = self.mu_matrix[t]
        return float(np.max(mu) - mu[chosen_arm])

    # =========================
    # validation
    # =========================
    def validate(self) -> None:
        T = self.contexts.shape[0]
        N = self.arm_features.shape[0]

        if self.mu_matrix.shape != (T, N):
            raise ValueError(
                f"mu_matrix shape mismatch: expected ({T}, {N}), got {self.mu_matrix.shape}"
            )

        if self.contexts.shape[1] != self.context_dim:
            raise ValueError("context_dim mismatch")

        if self.arm_features.shape[1] != self.arm_dim:
            raise ValueError("arm_dim mismatch")

        if self.selected_user_ids.shape[0] != N:
            raise ValueError("selected_user_ids mismatch")

        if self.movie_ids.shape[0] != T:
            raise ValueError("movie_ids mismatch")