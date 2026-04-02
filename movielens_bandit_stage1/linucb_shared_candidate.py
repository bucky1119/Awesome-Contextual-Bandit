"""Shared-parameter LinUCB for fixed-arm contextual bandits."""

from __future__ import annotations

import numpy as np


class LinUCBSharedCandidate:
    """
    Shared-parameter LinUCB for fixed-arm contextual bandits.

    Environment format:
    - arm_features: [N, d_arm]   (fixed across all rounds)
    - context:      [d_ctx]      (changes every round)

    For each arm i at round t, the algorithm builds:
        x_{t,i} = concat(arm_features[i], context_t)

    and scores it with one shared linear UCB model.
    """

    def __init__(
        self,
        arm_dim: int,
        context_dim: int,
        num_arms: int,
        alpha: float = 1.0,
        lambda_prior: float = 1.0,
        fit_intercept: bool = True,
        initial_pulls: int = 1,
    ) -> None:
        self.arm_dim = int(arm_dim)
        self.context_dim = int(context_dim)
        self.num_arms = int(num_arms)

        self.feature_dim = self.arm_dim + self.context_dim
        self.alpha = float(alpha)
        self.lambda_prior = float(lambda_prior)
        self.fit_intercept = bool(fit_intercept)
        self.initial_pulls = int(initial_pulls)

        self.d = self.feature_dim + (1 if self.fit_intercept else 0)
        self.t = 0

        eye = np.eye(self.d, dtype=np.float64)
        self.A_inv = (1.0 / self.lambda_prior) * eye
        self.b = np.zeros(self.d, dtype=np.float64)
        self.theta_hat = np.zeros(self.d, dtype=np.float64)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.feature_dim,):
            raise ValueError(
                f"joint feature shape must be ({self.feature_dim},), got {x.shape}"
            )
        if self.fit_intercept:
            return np.concatenate([x, np.array([1.0], dtype=np.float64)], axis=0)
        return x

    def _joint_feature(self, arm_feature: np.ndarray, context: np.ndarray) -> np.ndarray:
        arm_feature = np.asarray(arm_feature, dtype=np.float64)
        context = np.asarray(context, dtype=np.float64)

        if arm_feature.shape != (self.arm_dim,):
            raise ValueError(
                f"arm_feature shape must be ({self.arm_dim},), got {arm_feature.shape}"
            )
        if context.shape != (self.context_dim,):
            raise ValueError(
                f"context shape must be ({self.context_dim},), got {context.shape}"
            )

        return np.concatenate([arm_feature, context], axis=0)

    def action(self, arm_features: np.ndarray, context: np.ndarray) -> int:
        """
        Select one arm from all fixed arms.

        Args:
            arm_features: np.ndarray of shape [N, d_arm]
            context: np.ndarray of shape [d_ctx]

        Returns:
            chosen_arm: int in [0, N-1]
        """
        arm_features = np.asarray(arm_features, dtype=np.float64)
        context = np.asarray(context, dtype=np.float64)

        if arm_features.shape != (self.num_arms, self.arm_dim):
            raise ValueError(
                f"arm_features shape must be ({self.num_arms}, {self.arm_dim}), got {arm_features.shape}"
            )
        if context.shape != (self.context_dim,):
            raise ValueError(
                f"context shape must be ({self.context_dim},), got {context.shape}"
            )

        # Early exploration: round-robin over global arms
        if self.t < self.num_arms * self.initial_pulls:
            return int(self.t % self.num_arms)

        scores = np.zeros(self.num_arms, dtype=np.float64)
        for i in range(self.num_arms):
            x = self._augment(self._joint_feature(arm_features[i], context))
            mean = float(np.dot(self.theta_hat, x))
            unc = float(np.sqrt(np.dot(x, self.A_inv @ x)))
            scores[i] = mean + self.alpha * unc

        return int(np.argmax(scores))

    def update(self, arm_features: np.ndarray, context: np.ndarray, chosen_arm: int, reward: float) -> None:
        """
        Update the shared linear model with the chosen arm only.

        Args:
            arm_features: [N, d_arm]
            context: [d_ctx]
            chosen_arm: selected arm index
            reward: observed scalar reward
        """
        arm_features = np.asarray(arm_features, dtype=np.float64)
        context = np.asarray(context, dtype=np.float64)

        if arm_features.shape != (self.num_arms, self.arm_dim):
            raise ValueError(
                f"arm_features shape must be ({self.num_arms}, {self.arm_dim}), got {arm_features.shape}"
            )
        if context.shape != (self.context_dim,):
            raise ValueError(
                f"context shape must be ({self.context_dim},), got {context.shape}"
            )

        i = int(chosen_arm)
        if i < 0 or i >= self.num_arms:
            raise ValueError(f"chosen_arm out of range: {i}")

        x = self._augment(self._joint_feature(arm_features[i], context))
        reward = float(reward)
        self.t += 1

        # Sherman-Morrison rank-1 inverse update
        a_inv_x = self.A_inv @ x
        denom = 1.0 + float(np.dot(x, a_inv_x))
        self.A_inv -= np.outer(a_inv_x, a_inv_x) / denom

        self.b += reward * x
        self.theta_hat = self.A_inv @ self.b