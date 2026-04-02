"""Shared Neural-LinUCB for fixed-arm contextual bandits."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _SharedEncoderRegressor(nn.Module):
    """Encode a joint feature x into latent z and predict scalar reward."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.to_latent = nn.Linear(prev, latent_dim)
        self.reward_head = nn.Linear(latent_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.to_latent(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.reward_head(z).squeeze(1)


@dataclass
class NeuralLinUCBSharedCandidate:
    """
    Shared Neural-LinUCB for fixed-arm contextual bandits.

    Environment format:
    - arm_features: [N, d_arm]   (fixed across all rounds)
    - context:      [d_ctx]      (changes every round)

    For each arm i at round t:
        x_{t,i} = concat(arm_features[i], context_t)

    The neural network maps x_{t,i} -> z_{t,i},
    and one shared LinUCB head scores z_{t,i}.
    """

    arm_dim: int
    context_dim: int
    num_arms: int
    latent_dim: int = 32
    hidden_size: int = 100
    hidden_layers: int = 2
    alpha: float = 1.0
    lambda_prior: float = 1.0
    fit_intercept: bool = True
    initial_pulls: int = 1
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 5
    train_every: int = 100
    buffer_size: int = 10000
    dropout: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        self.arm_dim = int(self.arm_dim)
        self.context_dim = int(self.context_dim)
        self.num_arms = int(self.num_arms)

        self.feature_dim = self.arm_dim + self.context_dim
        self.latent_dim = int(self.latent_dim)
        self.hidden_size = int(self.hidden_size)
        self.hidden_layers = int(self.hidden_layers)
        self.alpha = float(self.alpha)
        self.lambda_prior = float(self.lambda_prior)
        self.fit_intercept = bool(self.fit_intercept)
        self.initial_pulls = int(self.initial_pulls)
        self.batch_size = int(self.batch_size)
        self.epochs = int(self.epochs)
        self.train_every = int(self.train_every)
        self.buffer_size = int(self.buffer_size)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        hidden_dims = [self.hidden_size for _ in range(self.hidden_layers)]
        self.net = _SharedEncoderRegressor(
            input_dim=self.feature_dim,
            hidden_dims=hidden_dims,
            latent_dim=self.latent_dim,
            dropout=float(self.dropout),
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=float(self.lr))

        self.lin_dim = self.latent_dim + (1 if self.fit_intercept else 0)
        eye = np.eye(self.lin_dim, dtype=np.float64)
        self.A_inv = (1.0 / self.lambda_prior) * eye
        self.b = np.zeros(self.lin_dim, dtype=np.float64)
        self.theta_hat = np.zeros(self.lin_dim, dtype=np.float64)

        self.t = 0
        self._buffer: Deque[Tuple[np.ndarray, float]] = deque(maxlen=self.buffer_size)

    # =========================
    # feature utilities
    # =========================
    def _joint_feature(self, arm_feature: np.ndarray, context: np.ndarray) -> np.ndarray:
        arm_feature = np.asarray(arm_feature, dtype=np.float32)
        context = np.asarray(context, dtype=np.float32)

        if arm_feature.shape != (self.arm_dim,):
            raise ValueError(
                f"arm_feature shape must be ({self.arm_dim},), got {arm_feature.shape}"
            )
        if context.shape != (self.context_dim,):
            raise ValueError(
                f"context shape must be ({self.context_dim},), got {context.shape}"
            )

        return np.concatenate([arm_feature, context], axis=0).astype(np.float32)

    def _joint_feature_batch(self, arm_features: np.ndarray, context: np.ndarray) -> np.ndarray:
        arm_features = np.asarray(arm_features, dtype=np.float32)
        context = np.asarray(context, dtype=np.float32)

        if arm_features.shape != (self.num_arms, self.arm_dim):
            raise ValueError(
                f"arm_features shape must be ({self.num_arms}, {self.arm_dim}), got {arm_features.shape}"
            )
        if context.shape != (self.context_dim,):
            raise ValueError(
                f"context shape must be ({self.context_dim},), got {context.shape}"
            )

        context_rep = np.repeat(context[None, :], self.num_arms, axis=0)
        return np.concatenate([arm_features, context_rep], axis=1).astype(np.float32)

    def _encode_np(self, x: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
            if x_t.ndim == 1:
                x_t = x_t.unsqueeze(0)
            z = self.net.encode(x_t).cpu().numpy()
        return z

    def _augment_latent(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (self.latent_dim,):
            raise ValueError(f"latent shape must be ({self.latent_dim},), got {z.shape}")
        if self.fit_intercept:
            return np.concatenate([z, np.array([1.0], dtype=np.float64)], axis=0)
        return z

    # =========================
    # action selection
    # =========================
    def action(self, arm_features: np.ndarray, context: np.ndarray) -> int:
        """
        Select one arm from all fixed arms.

        Args:
            arm_features: [N, d_arm]
            context: [d_ctx]

        Returns:
            chosen_arm: int in [0, N-1]
        """
        joint_x = self._joint_feature_batch(arm_features, context)  # [N, d]
        if self.t < self.num_arms * self.initial_pulls:
            return int(self.t % self.num_arms)

        z_all = self._encode_np(joint_x)  # [N, latent_dim]
        ucb_values = np.zeros(self.num_arms, dtype=np.float64)

        for i in range(self.num_arms):
            z_aug = self._augment_latent(z_all[i])
            mean = float(np.dot(self.theta_hat, z_aug))
            unc = float(np.sqrt(np.dot(z_aug, self.A_inv @ z_aug)))
            ucb_values[i] = mean + self.alpha * unc

        return int(np.argmax(ucb_values))

    # =========================
    # online update
    # =========================
    def update(self, arm_features: np.ndarray, context: np.ndarray, chosen_arm: int, reward: float) -> None:
        """
        Update shared latent LinUCB head using the selected arm only.
        """
        arm_features = np.asarray(arm_features, dtype=np.float32)
        context = np.asarray(context, dtype=np.float32)

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

        x = self._joint_feature(arm_features[i], context)
        r = float(reward)
        self.t += 1

        self._buffer.append((x.copy(), r))

        z = self._encode_np(x)[0]
        z_aug = self._augment_latent(z)

        a_inv_z = self.A_inv @ z_aug
        denom = 1.0 + float(np.dot(z_aug, a_inv_z))
        self.A_inv -= np.outer(a_inv_z, a_inv_z) / denom

        self.b += r * z_aug
        self.theta_hat = self.A_inv @ self.b

        if self.train_every > 0 and self.t % self.train_every == 0 and len(self._buffer) >= self.batch_size:
            self._train_network()
            self._rebuild_shared_head_from_buffer()

    # =========================
    # neural training
    # =========================
    def _train_network(self) -> None:
        self.net.train()

        x_arr = np.stack([x for x, _ in self._buffer], axis=0).astype(np.float32)
        r_arr = np.array([r for _, r in self._buffer], dtype=np.float32)
        n = x_arr.shape[0]

        for _ in range(self.epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                x_t = torch.from_numpy(x_arr[idx])
                r_t = torch.from_numpy(r_arr[idx])

                pred = self.net(x_t)
                loss = ((pred - r_t) ** 2).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    # =========================
    # rebuild linear head after encoder changes
    # =========================
    def _rebuild_shared_head_from_buffer(self) -> None:
        eye = np.eye(self.lin_dim, dtype=np.float64)
        self.A_inv = (1.0 / self.lambda_prior) * eye
        self.b = np.zeros(self.lin_dim, dtype=np.float64)
        self.theta_hat = np.zeros(self.lin_dim, dtype=np.float64)

        x_arr = np.stack([x for x, _ in self._buffer], axis=0).astype(np.float32)
        r_arr = np.array([r for _, r in self._buffer], dtype=np.float64)
        z_arr = self._encode_np(x_arr)

        for j in range(len(r_arr)):
            z_aug = self._augment_latent(z_arr[j])
            a_inv_z = self.A_inv @ z_aug
            denom = 1.0 + float(np.dot(z_aug, a_inv_z))
            self.A_inv -= np.outer(a_inv_z, a_inv_z) / denom
            self.b += float(r_arr[j]) * z_aug

        self.theta_hat = self.A_inv @ self.b