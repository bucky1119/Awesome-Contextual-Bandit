# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the License governing permissions and
# limitations under the License.
# ==============================================================================

"""Self-contained NeuralUCB (paper-style real-world approximation, PyTorch).

Closer to:
    Zhou, Li, Gu (2020).
    Neural Contextual Bandits with UCB-based Exploration.
    ICML 2020.

Key choices:
1. Shared scalar-output neural network.
2. Disjoint action-context representation:
       x^(a) = [0, ..., x, ..., 0] in R^(d * K)
3. Shared diagonal approximation of Z_t:
       Z_diag = lambda * 1 + sum_i g_i^2
4. UCB bonus:
       f(x^(a); theta) + alpha * sqrt(sum_j g_j^2 / Z_diag[j])
5. Periodic retraining:
       every 100 rounds starting from round 2000 by default
6. SGD training with paper-style regularization:
       0.5 * sum((f-r)^2) + 0.5 * m * lambda * ||theta - theta0||^2
7. Before every retraining, reset model to theta0.
8. After every retraining, rebuild Z_diag using the current retrained model
   so uncertainty statistics align with the current network.

Notes:
- This file is self-contained and does NOT depend on neural_bandit_model.py.
- It still uses ContextualDataset and BanditAlgorithm from your project.
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
from bandits.core.contextual_dataset import ContextualDataset


class SimpleNeuralRegressor(nn.Module):
  """Small MLP for scalar reward prediction."""

  def __init__(self, input_dim, layer_sizes, activation="relu", init_scale=None):
    super().__init__()

    self.input_dim = input_dim
    self.layer_sizes = layer_sizes
    self.activation_name = activation
    self.init_scale = init_scale

    layers = []
    cur_dim = input_dim

    for h in layer_sizes:
      if h <= 0:
        continue
      layers.append(nn.Linear(cur_dim, h))
      if activation == "relu":
        layers.append(nn.ReLU())
      elif activation == "tanh":
        layers.append(nn.Tanh())
      elif activation == "sigmoid":
        layers.append(nn.Sigmoid())
      else:
        raise ValueError("Unsupported activation: {}".format(activation))
      cur_dim = h

    layers.append(nn.Linear(cur_dim, 1))
    self.net = nn.Sequential(*layers)

    self._initialize_weights()

  def _initialize_weights(self):
    if self.init_scale is None:
      return

    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, -self.init_scale, self.init_scale)
        if module.bias is not None:
          nn.init.zeros_(module.bias)

  def forward(self, x):
    return self.net(x)


class NeuralUCBSampling(BanditAlgorithm):
  """NeuralUCB with shared diagonal Z and self-contained NN trainer."""

  def __init__(self, hparams, name="neural_ucb_diag_selfcontained"):
    self.name = name
    self.hparams = hparams

    self.num_actions = hparams["num_actions"]
    self.context_dim = hparams["context_dim"]

    # Exploration coefficient.
    self.alpha = hparams.get("alpha", 1.0)

    # lambda in paper
    self._lambda_prior = hparams.get("lambda_prior", 1.0)

    # Paper real-world setting:
    # update every 100 rounds starting from round 2000
    self.update_freq_nn = hparams.get(
        "training_freq_network",
        hparams.get("training_freq", 100)
    )
    self.training_starts_at = hparams.get("training_starts_at", 2000)
    self.num_epochs = hparams.get("training_epochs", 1000)

    self.initial_pulls = hparams.get("initial_pulls", 2)

    # Network settings
    self.layer_sizes = hparams.get("layer_sizes", [100])
    self.activation = hparams.get("activation", "relu")
    self.learning_rate = hparams.get("initial_lr", 0.01)
    self.batch_size = hparams.get("batch_size", 500)
    self.init_scale = hparams.get("init_scale", None)
    self.verbose = hparams.get("verbose", False)

    # For paper's regularizer (m * lambda / 2) ||theta - theta0||^2
    self.hidden_size = self.layer_sizes[0] if len(self.layer_sizes) > 0 else 1

    self.t = 0

    # ===== Disjoint representation =====
    # x^(a) in R^(d*K), not [x; one_hot(a)]
    self.aug_dim = self.context_dim * self.num_actions

    # Store augmented contexts and scalar rewards
    self.data_h = ContextualDataset(
        np.empty((0, self.aug_dim), dtype=np.float32),
        np.empty((0, 1), dtype=np.float32)
    )

    # Cache chosen augmented contexts so Z_diag can be rebuilt after retraining
    self.chosen_history = []

    # Shared scalar-output network
    self.model = SimpleNeuralRegressor(
        input_dim=self.aug_dim,
        layer_sizes=self.layer_sizes,
        activation=self.activation,
        init_scale=self.init_scale
    )

    # Save theta0 for paper-style regularization
    self.theta0 = {
        k: v.detach().clone()
        for k, v in self.model.state_dict().items()
    }

    self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    # Total parameter dimension p
    self.total_param_dim = sum(param.numel() for param in self.model.parameters())

    # Shared diagonal approximation of Z_t
    # Z_diag[j] = lambda + sum_i g_i[j]^2
    self.Z_diag = self._lambda_prior * np.ones(self.total_param_dim, dtype=np.float64)

    if self.verbose:
      print("[{}] aug_dim={}, total_param_dim={}".format(
          self.name, self.aug_dim, self.total_param_dim))

  def _disjoint_context(self, context, arm_idx):
    """Construct x^(a) in R^(d*K)."""
    context = np.asarray(context, dtype=np.float32).reshape(-1)
    aug = np.zeros(self.aug_dim, dtype=np.float32)
    start = arm_idx * self.context_dim
    end = start + self.context_dim
    aug[start:end] = context
    return aug

  def _tensorize_context(self, context, arm_idx):
    aug_ctx = self._disjoint_context(context, arm_idx)
    return torch.tensor(aug_ctx, dtype=torch.float32).unsqueeze(0)

  def _predict(self, context, action_idx):
    """Predict scalar reward for (context, action)."""
    self.model.eval()
    with torch.no_grad():
      x = self._tensorize_context(context, action_idx)
      pred = self.model(x).item()
    return pred

  def _get_gradient(self, context, action_idx):
    """Compute g(x^(a)) = ∇_theta f / sqrt(m)."""
    was_training = self.model.training
    self.model.eval()

    x = self._tensorize_context(context, action_idx)

    self.model.zero_grad()
    out = self.model(x)
    out[0, 0].backward()

    grads = []
    for param in self.model.parameters():
      if param.grad is None:
        grads.append(np.zeros(param.numel(), dtype=np.float64))
      else:
        grads.append(param.grad.detach().cpu().numpy().reshape(-1).astype(np.float64))

    if was_training:
      self.model.train()

    grad = np.concatenate(grads, axis=0) / np.sqrt(self.hidden_size)
    return grad

  def _get_gradient_from_aug_context(self, aug_ctx):
    """Compute gradient feature from an already-augmented disjoint context."""
    was_training = self.model.training
    self.model.eval()

    x = torch.tensor(aug_ctx, dtype=torch.float32).unsqueeze(0)

    self.model.zero_grad()
    out = self.model(x)
    out[0, 0].backward()

    grads = []
    for param in self.model.parameters():
      if param.grad is None:
        grads.append(np.zeros(param.numel(), dtype=np.float64))
      else:
        grads.append(param.grad.detach().cpu().numpy().reshape(-1).astype(np.float64))

    if was_training:
      self.model.train()

    grad = np.concatenate(grads, axis=0) / np.sqrt(self.hidden_size)
    return grad

  def _rebuild_Z_diag(self):
    """Rebuild shared diagonal Z using the current retrained model."""
    z_diag = self._lambda_prior * np.ones(self.total_param_dim, dtype=np.float64)

    for aug_ctx in self.chosen_history:
      g = self._get_gradient_from_aug_context(aug_ctx)
      z_diag += g ** 2

    self.Z_diag = z_diag

  def _parameter_deviation_penalty(self):
    """||theta - theta0||^2."""
    penalty = 0.0
    current_state = self.model.state_dict()
    for name, param in current_state.items():
      theta0_param = self.theta0[name].to(param.device)
      penalty = penalty + torch.sum((param - theta0_param) ** 2)
    return penalty

  def _train_one_step(self, contexts, rewards):
    """One SGD step on a minibatch."""
    self.optimizer.zero_grad()

    preds = self.model(contexts).squeeze(-1)
    rewards = rewards.view(-1)

    # More faithful to the paper than batch mean
    data_loss = 0.5 * torch.sum((preds - rewards) ** 2)
    reg_loss = 0.5 * self.hidden_size * self._lambda_prior * self._parameter_deviation_penalty()
    loss = data_loss + reg_loss

    loss.backward()
    self.optimizer.step()

    return float(loss.item())

  def _train_model(self, dataset, num_steps):
    """Periodic retraining on full history via minibatch SGD."""
    if len(dataset) == 0:
      return

    # Important: reset to theta0 before every retraining round
    self.model.load_state_dict(copy.deepcopy(self.theta0))
    self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    self.model.train()

    n = len(dataset)
    replace = n < self.batch_size

    for step in range(num_steps):
      batch_indices = np.random.choice(n, self.batch_size, replace=replace)

      contexts_batch = []
      rewards_batch = []

      for idx in batch_indices:
        context, reward = dataset[idx]

        if isinstance(context, torch.Tensor):
          ctx = context.float()
        else:
          ctx = torch.tensor(context, dtype=torch.float32)

        if isinstance(reward, torch.Tensor):
          if reward.dim() == 0:
            r = float(reward.item())
          else:
            r = float(reward.reshape(-1)[0].item())
        elif isinstance(reward, (np.ndarray, list)):
          r = float(np.asarray(reward).reshape(-1)[0])
        else:
          r = float(reward)

        contexts_batch.append(ctx)
        rewards_batch.append(r)

      contexts = torch.stack(contexts_batch, dim=0)
      rewards = torch.tensor(rewards_batch, dtype=torch.float32)

      loss_val = self._train_one_step(contexts, rewards)

      if self.verbose and step % 100 == 0:
        print("[{}] train step={} loss={:.6f}".format(
            self.name, step, loss_val))

    self.model.eval()

    # # Important: after retraining, align Z_diag with the current model
    # self._rebuild_Z_diag()

  def action(self, context):
    """Select action with UCB."""
    if self.t < self.num_actions * self.initial_pulls:
      return self.t % self.num_actions

    ucb_values = np.zeros(self.num_actions, dtype=np.float64)

    for a in range(self.num_actions):
      pred = self._predict(context, a)
      g = self._get_gradient(context, a)

      # Diagonal Z approximation:
      # g^T Z^{-1} g ≈ sum_j g_j^2 / Z_diag[j]
      quad = float(np.sum((g ** 2) / self.Z_diag))
      quad = max(quad, 0.0)

      confidence = self.alpha * np.sqrt(quad)
      ucb_values[a] = pred + confidence

    return int(np.argmax(ucb_values))

  def update(self, context, action, reward):
    """Update history, diagonal Z, and periodically retrain NN."""
    self.t += 1

    aug_ctx = self._disjoint_context(context, action)
    self.data_h.add(aug_ctx, 0, reward)
    self.chosen_history.append(aug_ctx.copy())

    # Update shared diagonal Z using chosen action only
    g = self._get_gradient(context, action)
    self.Z_diag += g ** 2

    if self.t >= self.training_starts_at and self.t % self.update_freq_nn == 0:
      self._train_model(self.data_h, self.num_epochs)

  @property
  def lambda_prior(self):
    return self._lambda_prior