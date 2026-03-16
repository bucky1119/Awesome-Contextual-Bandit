# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
# ==============================================================================

"""Interconnected Neural-Linear UCB (InlUCB) algorithm (PyTorch version).

Reference:
    Chen, Xie, Liu, Zhao (2022).
    Interconnected Neural Linear Contextual Bandits with UCB Exploration.

Paper idea:
    - Online phase: fix representation f, run LinUCB on top of latent features.
    - Offline phase: fix the linear weight theta, train only f by MSE:
          L_D(f; theta) = E[(f(x)^T theta - r)^2]

Important:
    This implementation follows the paper's setting where each round observes
    one context per action:
        contexts.shape == (num_actions, context_dim)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bandits.core.bandit_algorithm import BanditAlgorithm


class InlUCBDataset(object):
  """Simple offline dataset for InlUCB.

  Stores tuples of:
      (chosen_context, chosen_action, observed_reward)

  The action is stored for bookkeeping / analysis, although the paper's offline
  loss only needs (x, r) because theta is shared across actions.
  """

  def __init__(self, context_dim):
    self.context_dim = context_dim
    self.contexts = []
    self.actions = []
    self.rewards = []

  def add(self, context, action, reward):
    self.contexts.append(np.asarray(context, dtype=np.float32))
    self.actions.append(int(action))
    self.rewards.append(float(reward))

  def __len__(self):
    return len(self.contexts)

  def get_all(self):
    if len(self.contexts) == 0:
      return (
          np.empty((0, self.context_dim), dtype=np.float32),
          np.empty((0,), dtype=np.int64),
          np.empty((0,), dtype=np.float32),
      )
    return (
        np.asarray(self.contexts, dtype=np.float32),
        np.asarray(self.actions, dtype=np.int64),
        np.asarray(self.rewards, dtype=np.float32),
    )


class InlUCBNetwork(nn.Module):
  """Feature network f(x) that outputs latent representation z in R^p."""

  def __init__(self, input_dim, layer_sizes, activation="relu",
               use_dropout=False, dropout_rate=0.0, layer_norm=False):
    super().__init__()

    if len(layer_sizes) == 0:
      raise ValueError("layer_sizes must be non-empty")

    modules = []
    prev_dim = input_dim

    for i, hidden_dim in enumerate(layer_sizes):
      modules.append(nn.Linear(prev_dim, hidden_dim))

      if layer_norm:
        modules.append(nn.LayerNorm(hidden_dim))

      if i < len(layer_sizes) - 1:
        modules.append(self._get_activation(activation))

        if use_dropout:
          modules.append(nn.Dropout(dropout_rate))

      prev_dim = hidden_dim

    self.feature_extractor = nn.Sequential(*modules)

  def _get_activation(self, activation):
    activation = activation.lower()
    if activation == "relu":
      return nn.ReLU()
    if activation == "tanh":
      return nn.Tanh()
    if activation == "sigmoid":
      return nn.Sigmoid()
    if activation == "elu":
      return nn.ELU()
    raise ValueError(f"Unsupported activation: {activation}")

  def forward(self, x):
    return self.feature_extractor(x)


class InlUCBSampling(BanditAlgorithm):
  """InlUCB: shared linear UCB head + offline representation learning.

  Paper-aligned structure:
    - Online exploration:
        Fix f_{n-1}, run LinUCB on z = f_{n-1}(x)
    - Offline representation learning:
        Fix theta_n, update only f_n on the accumulated offline dataset

  Expected action() input:
      contexts: np.ndarray of shape (num_actions, context_dim)

  Expected update() input:
      chosen_context: np.ndarray of shape (context_dim,)
      action: int
      reward: float
  """

  def __init__(self, hparams, name="inlucb"):
    self.name = name
    self.hparams = hparams

    self.num_actions = hparams["num_actions"]
    self.context_dim = hparams["context_dim"]
    self.layer_sizes = hparams["layer_sizes"]
    self.latent_dim = self.layer_sizes[-1]

    # UCB / ridge params
    self.alpha = hparams.get("alpha", 1.0)
    self._lambda_prior = hparams.get("lambda_prior", 1.0)

    # Paper uses iterations:
    #   each iteration has T online steps, then one offline update
    self.online_horizon = hparams.get("online_horizon", 100)
    self.offline_epochs = hparams.get("offline_epochs", 100)

    # Optim params for offline representation learning
    self.initial_lr = hparams.get("initial_lr", 1e-3)
    self.batch_size = hparams.get("batch_size", 64)
    self.weight_decay = hparams.get("weight_decay", 0.0)

    self.use_dropout = hparams.get("use_dropout", False)
    self.dropout_rate = hparams.get("dropout_rate", 0.0)
    self.layer_norm = hparams.get("layer_norm", False)
    self.activation = hparams.get("activation", "relu")
    self.verbose = hparams.get("verbose", False)

    # Optional warm start: round-robin initially
    self.initial_pulls = hparams.get("initial_pulls", 0)

    # Global counters
    self.t = 0                 # total online steps
    self.iteration = 1         # paper's n
    self.iteration_step = 0    # step index inside current iteration

    # Shared LinUCB statistics — paper Algorithm 1: A' = Ip, b' = 0p (fixed anchors).
    # At the start of every online phase: A ← Ip, b ← 0.  Reset after offline update.
    self.A = self._lambda_prior * np.eye(self.latent_dim, dtype=np.float32)
    self.A_inv = (1.0 / self._lambda_prior) * np.eye(self.latent_dim, dtype=np.float32)
    self.b = np.zeros(self.latent_dim, dtype=np.float32)
    self.theta = np.zeros(self.latent_dim, dtype=np.float32)

    # Offline dataset D_n
    self.data_h = InlUCBDataset(self.context_dim)

    # Online samples collected within current iteration
    self._current_iteration_contexts = []
    self._current_iteration_actions = []
    self._current_iteration_rewards = []

    # Feature network f(x)
    self.network = InlUCBNetwork(
        input_dim=self.context_dim,
        layer_sizes=self.layer_sizes,
        activation=self.activation,
        use_dropout=self.use_dropout,
        dropout_rate=self.dropout_rate,
        layer_norm=self.layer_norm,
    )

  def _encode(self, contexts):
    """Compute latent representations z = f(x).

    Args:
      contexts: shape (K, d) or (d,)

    Returns:
      latent representations with shape (K, p) or (p,)
    """
    single = False
    x = np.asarray(contexts, dtype=np.float32)
    if x.ndim == 1:
      x = x[None, :]
      single = True

    x_tensor = torch.tensor(x, dtype=torch.float32)
    self.network.eval()
    with torch.no_grad():
      z = self.network(x_tensor).cpu().numpy()

    return z[0] if single else z

  def _sherman_morrison_update(self, z, reward):
    """Incrementally update A_inv, A, b, theta."""
    z = np.asarray(z, dtype=np.float32)

    A_inv_z = np.dot(self.A_inv, z)
    denom = 1.0 + np.dot(z, A_inv_z)

    self.A_inv -= np.outer(A_inv_z, A_inv_z) / denom
    self.A += np.outer(z, z)
    self.b += reward * z
    self.theta = np.dot(self.A_inv, self.b)

  def action(self, contexts):
    """Select action.

    Accepts two input formats:
      (a) Single shared context, shape (context_dim,)  — standard benchmark protocol.
          The same context is broadcast to all arms.
      (b) Per-arm contexts, shape (num_actions, context_dim) — paper protocol.

    Args:
      contexts: np.ndarray of shape (context_dim,) or (num_actions, context_dim)

    Returns:
      action index (int)
    """
    contexts = np.asarray(contexts, dtype=np.float32)

    # Broadcast single context to all arms
    if contexts.ndim == 1:
      if contexts.shape[0] != self.context_dim:
        raise ValueError(
            f"Single context must have shape ({self.context_dim},), got {contexts.shape}"
        )
      contexts = np.tile(contexts[None, :], (self.num_actions, 1))  # (K, d)
    elif contexts.shape != (self.num_actions, self.context_dim):
      raise ValueError(
          f"contexts must have shape ({self.num_actions}, {self.context_dim}) "
          f"or ({self.context_dim},), got {contexts.shape}"
      )

    # Optional warm-start round robin
    if self.t < self.num_actions * self.initial_pulls:
      return self.t % self.num_actions

    z_all = self._encode(contexts)  # (K, p)

    # theta is shared across all actions
    mean = np.dot(z_all, self.theta)  # (K,)
    quad = np.einsum("bi,ij,bj->b", z_all, self.A_inv, z_all)
    bonus = self.alpha * np.sqrt(np.maximum(quad, 0.0))
    ucb = mean + bonus

    return int(np.argmax(ucb))

  def update(self, context, action, reward):
    """Update online shared linear UCB statistics and trigger offline learning.

    Args:
      context: chosen action context, shape (context_dim,)
      action: chosen action index
      reward: scalar reward
    """
    self.t += 1
    self.iteration_step += 1

    context = np.asarray(context, dtype=np.float32)

    # Save to offline dataset
    self.data_h.add(context, action, reward)

    # Save current iteration samples
    self._current_iteration_contexts.append(context.copy())
    self._current_iteration_actions.append(int(action))
    self._current_iteration_rewards.append(float(reward))

    # Online exploration update: only shared linear head stats
    z = self._encode(context)
    self._sherman_morrison_update(z, reward)

    # End of one online iteration -> run offline representation learning
    if self.iteration_step >= self.online_horizon:
      self._end_iteration_and_train_representation()

  def _end_iteration_and_train_representation(self):
    """Finish one online iteration and run offline representation learning.

    Follows Algorithm 1 of the paper exactly:
      1) End of T online steps -> theta_n = A^{-1}b  (already accumulated in self.theta)
      2) Dn already updated incrementally in update()
      3) Fix theta_n, train f_n on Dn minimising MSE: (f(x)^T theta_n - r)^2
      4) Reset A <- Ip, b <- 0  (paper line 3: A <- A' = Ip, b <- b' = 0)
         Statistics from iteration n are discarded; f_n is now a *better*
         representation, so the next online phase starts clean.
    """
    if len(self.data_h) == 0:
      self.iteration += 1
      self.iteration_step = 0
      return

    # Freeze theta_n for offline optimisation
    theta_fixed = torch.tensor(self.theta.copy(), dtype=torch.float32)

    self._train_representation(theta_fixed)

    # Paper Algorithm 1, line 3: every online phase begins with A = Ip, b = 0.
    # A' and b' in the pseudocode are ALWAYS Ip and 0 (never updated).
    self.A = self._lambda_prior * np.eye(self.latent_dim, dtype=np.float32)
    self.A_inv = (1.0 / self._lambda_prior) * np.eye(self.latent_dim, dtype=np.float32)
    self.b = np.zeros(self.latent_dim, dtype=np.float32)
    self.theta = np.zeros(self.latent_dim, dtype=np.float32)

    # Clear per-iteration sample cache
    self._current_iteration_contexts = []
    self._current_iteration_actions = []
    self._current_iteration_rewards = []

    self.iteration += 1
    self.iteration_step = 0

  def _train_representation(self, theta_fixed):
    """Train only f while freezing theta.

    Minimize:
        E[(f(x)^T theta_fixed - r)^2]
    """
    contexts, _, rewards = self.data_h.get_all()
    if len(contexts) == 0:
      return

    dataset_size = len(contexts)
    x_tensor = torch.tensor(contexts, dtype=torch.float32)
    y_tensor = torch.tensor(rewards, dtype=torch.float32)

    optimizer = optim.Adam(
        self.network.parameters(),
        lr=self.initial_lr,
        weight_decay=self.weight_decay,
    )
    mse_loss = nn.MSELoss()

    self.network.train()

    for epoch in range(self.offline_epochs):
      perm = torch.randperm(dataset_size)
      epoch_loss = 0.0

      for start in range(0, dataset_size, self.batch_size):
        idx = perm[start:start + self.batch_size]
        xb = x_tensor[idx]
        yb = y_tensor[idx]

        optimizer.zero_grad()

        z = self.network(xb)                    # (B, p)
        pred = torch.matmul(z, theta_fixed)     # (B,)

        loss = mse_loss(pred, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(idx)

      if self.verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
        avg_loss = epoch_loss / dataset_size
        print(
            f"[{self.name}] iteration={self.iteration} "
            f"offline_epoch={epoch + 1}/{self.offline_epochs} "
            f"loss={avg_loss:.6f}"
        )

  def force_offline_train(self):
    """Manually finish the current iteration and run offline update.

    Useful when training ends before online_horizon is reached.
    """
    if self.iteration_step > 0:
      self._end_iteration_and_train_representation()

  @property
  def lambda_prior(self):
    return self._lambda_prior