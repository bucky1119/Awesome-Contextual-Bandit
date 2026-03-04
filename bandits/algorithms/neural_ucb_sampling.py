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
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Neural UCB algorithm for contextual bandits (PyTorch version).

Implements the NeuralUCB algorithm which uses a neural network to predict
rewards and constructs UCB exploration bonuses based on the gradient of the
network output with respect to the network parameters.

Reference:
    Zhou, Li, Gu (2020).
    Neural Contextual Bandits with UCB-based Exploration.
    ICML 2020.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset import ContextualDataset
from bandits.algorithms.neural_bandit_model import NeuralBanditModel


class NeuralUCBSampling(BanditAlgorithm):
  """Neural UCB algorithm with gradient-based exploration bonus.

  Uses a neural network f(x; theta) to predict rewards for each action.
  The exploration bonus is computed using the gradient of the network output:
      UCB_a(x) = f_a(x; theta) + alpha * sqrt( g_a(x)^T Z_a^{-1} g_a(x) )
  where g_a(x) = grad_theta f_a(x; theta) is the gradient of the network
  output for action a with respect to all parameters, and Z_a is a
  regularized gram matrix of gradients: Z_a = lambda*I + sum g_a(x_i) g_a(x_i)^T.

  For computational efficiency, Z_a^{-1} is maintained incrementally using
  the Sherman-Morrison formula.

  Reference:
      Zhou, Li, Gu (2020).
      Neural Contextual Bandits with UCB-based Exploration.
      ICML 2020.
  """

  def __init__(self, hparams, name="neural_ucb"):
    """Initialize Neural UCB.

    Args:
      hparams: Dictionary of hyper-parameters containing:
        - context_dim: Dimension of context vectors.
        - num_actions: Number of arms/actions.
        - layer_sizes: List of hidden layer sizes.
        - activation: Activation function name.
        - initial_lr: Learning rate.
        - batch_size: Batch size for NN training.
        - init_scale: Weight initialization scale.
        - use_dropout: Whether to use dropout.
        - dropout_rate: Dropout rate.
        - layer_norm: Whether to use layer normalization.
        - verbose: Whether to print training info.
        - alpha: UCB exploration coefficient (default: 1.0).
        - lambda_prior: Regularization for the gradient gram matrix (default: 1.0).
        - training_freq: Frequency of NN retraining (default: 100).
        - training_freq_network: Deprecated alias for training_freq.
        - training_epochs: Number of NN training epochs (default: 100).
        - initial_pulls: Number of initial round-robin pulls per action (default: 2).
    """
    self.name = name
    self.hparams = hparams
    self.num_actions = hparams["num_actions"]
    self.context_dim = hparams["context_dim"]
    self.alpha = hparams.get("alpha", 1.0)  # UCB 探索系数
    self._lambda_prior = hparams.get("lambda_prior", 1.0)  # 梯度 Gram 矩阵正则化参数
    self.update_freq_nn = hparams.get("training_freq_network", hparams.get("training_freq", 100))  # NN 训练频率
    self.num_epochs = hparams.get("training_epochs", 100)  # NN 训练轮数
    self.initial_pulls = hparams.get("initial_pulls", 2)  # 初始轮询次数
    self.t = 0  # 全局时间步

    # 数据集用于存储原始 (context, action, reward) 数据
    self.data_h = ContextualDataset(
        np.empty((0, self.context_dim)),
        np.empty((0, self.num_actions))
    )

    # 建立神经网络模型
    self.bnn = NeuralBanditModel(hparams, name=f"{name}-bnn")

    # 计算网络总参数量 p（用于梯度向量维度）
    self.p = sum(param.numel() for param in self.bnn.parameters())

    # 为每个动作维护 Z_a^{-1}（p x p 矩阵太大时退化到对角近似）
    # 为了计算效率，使用对角近似：Z_a ≈ lambda*I + diag(sum g_a(x_i)^2)
    # 这样 Z_a^{-1} 也是对角的，只需存储 p 维向量
    self.Z_inv_diag = [
        (1.0 / self._lambda_prior) * np.ones(self.p)
        for _ in range(self.num_actions)
    ]

  def _get_gradient(self, context, action_idx):
    """Compute the gradient of the network output for a given action w.r.t. parameters.

    Args:
      context: Context vector of shape (context_dim,).
      action_idx: Index of the action to compute gradient for.

    Returns:
      grad: Flattened gradient vector of shape (p,).
    """
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
    context_tensor.requires_grad_(False)

    # 前向传播
    self.bnn.zero_grad()
    output = self.bnn(context_tensor)  # shape: (1, num_actions)

    # 对第 action_idx 个输出求参数梯度
    output[0, action_idx].backward()

    # 拼接所有参数的梯度为一个向量
    grads = []
    for param in self.bnn.parameters():
      if param.grad is not None:
        grads.append(param.grad.detach().cpu().numpy().flatten())
      else:
        grads.append(np.zeros(param.numel()))

    return np.concatenate(grads)

  def action(self, context):
    """Selects the action with the highest Neural UCB index.

    Args:
      context: Context for which the action needs to be chosen.

    Returns:
      action: Selected action index.
    """

    # 初始阶段：轮流选择每个动作初始次数
    if self.t < self.num_actions * self.initial_pulls:
      return self.t % self.num_actions

    # 计算每个动作的 UCB 值
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

    ucb_values = np.zeros(self.num_actions)
    with torch.no_grad():
      predictions = self.bnn(context_tensor).numpy().squeeze(0)  # 网络预测奖励

    for a in range(self.num_actions):
      # 计算梯度
      g = self._get_gradient(context, a)
      # 对角近似的置信宽度: alpha * sqrt(g^T Z_a^{-1} g) ≈ alpha * sqrt(sum(g^2 * z_inv_diag))
      confidence = self.alpha * np.sqrt(np.sum(g ** 2 * self.Z_inv_diag[a]))
      ucb_values[a] = predictions[a] + confidence

    return int(np.argmax(ucb_values))

  def update(self, context, action, reward):
    """Updates the model and the gradient gram matrix.

    Args:
      context: Last observed context.
      action: Last observed action.
      reward: Last observed reward.
    """

    self.t += 1
    self.data_h.add(context, action, reward)

    # 计算当前 (context, action) 的梯度并更新 Z_a^{-1}（对角近似）
    g = self._get_gradient(context, action)
    # 使用对角 Sherman-Morrison 近似：
    # Z_new_diag = Z_old_diag + g^2
    # Z_inv_new_diag ≈ Z_inv_old_diag - (Z_inv_old_diag * g^2 * Z_inv_old_diag) / (1 + g^2 * Z_inv_old_diag)
    g_sq = g ** 2
    z_inv = self.Z_inv_diag[action]
    denom = 1.0 + g_sq * z_inv
    self.Z_inv_diag[action] = z_inv - (z_inv * g_sq * z_inv) / denom

    # 定期重训神经网络
    if self.t % self.update_freq_nn == 0:
      self.bnn.train_model(self.data_h, self.num_epochs)

  @property
  def lambda_prior(self):
    return self._lambda_prior
