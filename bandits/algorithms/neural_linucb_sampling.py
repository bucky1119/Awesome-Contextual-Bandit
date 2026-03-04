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

"""Neural LinUCB algorithm for contextual bandits (PyTorch version).

Combines a deep neural network representation with LinUCB-style UCB exploration
on the last hidden layer features. This is analogous to NeuralLinearPosteriorSampling
but replaces Thompson Sampling with UCB-based exploration.

Reference:
    Xu, Hsu, Maleki (2020).
    Neural Contextual Bandits with Deep Representation and Shallow Exploration.
    (Also known as Neural LinUCB / NeuralLinear-UCB.)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset import ContextualDataset
from bandits.algorithms.neural_bandit_model import NeuralBanditModel


class NeuralLinUCBSampling(BanditAlgorithm):
  """Neural LinUCB: deep representation + LinUCB exploration on last layer.

  Uses the last hidden layer of a neural network as a learned feature map
  phi(x), then runs LinUCB on these features:
      a_t = argmax_a [ phi(x)^T theta_hat_a + alpha * sqrt( phi(x)^T A_a^{-1} phi(x) ) ]

  The neural network is periodically retrained on all collected data, and the
  latent representations are refreshed accordingly.

  Reference:
      Xu, Hsu, Maleki (2020).
      Neural Contextual Bandits with Deep Representation and Shallow Exploration.
  """

  def __init__(self, hparams, name="neural_linucb"):
    """Initialize Neural LinUCB.

    Args:
      hparams: Dictionary of hyper-parameters containing:
        - context_dim: Dimension of context vectors.
        - num_actions: Number of arms/actions.
        - layer_sizes: List of hidden layer sizes (last element is latent dim).
        - activation: Activation function name.
        - initial_lr: Learning rate.
        - batch_size: Batch size for NN training.
        - init_scale: Weight initialization scale.
        - use_dropout: Whether to use dropout.
        - dropout_rate: Dropout rate.
        - layer_norm: Whether to use layer normalization.
        - verbose: Whether to print training info.
        - alpha: UCB exploration coefficient (default: 1.0).
        - lambda_prior: Ridge regularization parameter (default: 0.25).
        - training_freq: Frequency of LinUCB parameter update (default: 100).
        - training_freq_network: Frequency of NN retraining (default: 100).
        - training_epochs: Number of NN training epochs (default: 100).
        - initial_pulls: Number of initial round-robin pulls per action (default: 2).
    """
    self.name = name
    self.hparams = hparams
    self.num_actions = hparams["num_actions"]
    self.context_dim = hparams["context_dim"]
    self.latent_dim = hparams["layer_sizes"][-1]  # 最后一层隐藏层维度作为特征维度
    self.alpha = hparams.get("alpha", 1.0)  # UCB 探索系数
    self._lambda_prior = hparams.get("lambda_prior", 0.25)  # 岭回归正则化参数
    self.update_freq_lr = hparams.get("training_freq", 100)  # 线性回归更新频率
    self.update_freq_nn = hparams.get("training_freq_network", 100)  # NN 训练频率
    self.num_epochs = hparams.get("training_epochs", 100)  # NN 训练轮数
    self.initial_pulls = hparams.get("initial_pulls", 2)  # 初始轮询次数
    self.t = 0  # 全局时间步

    # 为每个动作初始化 LinUCB 参数
    self.A = [self._lambda_prior * np.eye(self.latent_dim) for _ in range(self.num_actions)]  # 精度矩阵
    self.A_inv = [(1.0 / self._lambda_prior) * np.eye(self.latent_dim) for _ in range(self.num_actions)]  # 精度矩阵的逆
    self.b_vec = [np.zeros(self.latent_dim) for _ in range(self.num_actions)]  # 奖励加权特征累加向量
    self.theta_hat = [np.zeros(self.latent_dim) for _ in range(self.num_actions)]  # 参数估计

    # 数据集
    self.data_h = ContextualDataset(
        np.empty((0, self.context_dim)),
        np.empty((0, self.num_actions))
    )  # 原始上下文数据集
    self.latent_h = ContextualDataset(
        np.empty((0, self.latent_dim)),
        np.empty((0, self.num_actions))
    )  # 潜在表示数据集

    # 神经网络模型
    self.bnn = NeuralBanditModel(hparams, name=f"{name}-bnn")

  def action(self, context):
    """Selects the action with the highest Neural LinUCB index.

    Args:
      context: Context for which the action needs to be chosen.

    Returns:
      action: Selected action index.
    """

    # 初始阶段：轮流选择动作
    if self.t < self.num_actions * self.initial_pulls:
      return self.t % self.num_actions

    # 计算当前上下文的最后一层表示
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
      z_context = self.bnn.network[:-1](context_tensor).numpy().squeeze(0)  # 排除最后一层输出层

    # 计算每个动作的 LinUCB 值
    ucb_values = np.zeros(self.num_actions)
    for a in range(self.num_actions):
      # 预测均值: z^T theta_hat_a
      pred = np.dot(self.theta_hat[a], z_context)
      # 置信宽度: alpha * sqrt(z^T A_a^{-1} z)
      confidence = self.alpha * np.sqrt(np.dot(z_context, np.dot(self.A_inv[a], z_context)))
      ucb_values[a] = pred + confidence

    return int(np.argmax(ucb_values))

  def update(self, context, action, reward):
    """Updates the neural network and LinUCB parameters.

    Args:
      context: Last observed context.
      action: Last observed action.
      reward: Last observed reward.
    """

    self.t += 1
    self.data_h.add(context, action, reward)

    # 计算当前上下文的潜在表示
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
      z_context = self.bnn.network[:-1](context_tensor).numpy().squeeze(0)
    self.latent_h.add(z_context, action, reward)

    # 增量更新 LinUCB 的 A_inv（Sherman-Morrison 公式）
    A_inv_z = np.dot(self.A_inv[action], z_context)
    denom = 1.0 + np.dot(z_context, A_inv_z)
    self.A_inv[action] -= np.outer(A_inv_z, A_inv_z) / denom
    self.A[action] += np.outer(z_context, z_context)
    self.b_vec[action] += reward * z_context
    self.theta_hat[action] = np.dot(self.A_inv[action], self.b_vec[action])

    # 定期重训神经网络
    if self.t % self.update_freq_nn == 0:
      self.bnn.train_model(self.data_h, self.num_epochs)

      # 重训后刷新所有潜在表示
      all_contexts = self.data_h.contexts.numpy()
      with torch.no_grad():
        new_z = self.bnn.network[:-1](
            torch.tensor(all_contexts, dtype=torch.float32)
        ).numpy()
      self.latent_h.contexts = torch.tensor(new_z, dtype=torch.float32)

      # 用刷新后的潜在表示完全重建 LinUCB 参数
      self._rebuild_linucb_params()

  def _rebuild_linucb_params(self):
    """Rebuild all LinUCB parameters from scratch using current latent representations."""

    # 重新初始化所有动作的参数
    self.A = [self._lambda_prior * np.eye(self.latent_dim) for _ in range(self.num_actions)]
    self.A_inv = [(1.0 / self._lambda_prior) * np.eye(self.latent_dim) for _ in range(self.num_actions)]
    self.b_vec = [np.zeros(self.latent_dim) for _ in range(self.num_actions)]
    self.theta_hat = [np.zeros(self.latent_dim) for _ in range(self.num_actions)]

    # 遍历每个动作，重建参数
    for action_v in range(self.num_actions):
      z, y = self.latent_h.get_batch_for_action(action_v)
      if len(z) == 0:
        continue
      # A_a = lambda * I + Z^T Z
      s = np.dot(z.T, z)
      self.A[action_v] = s + self._lambda_prior * np.eye(self.latent_dim)
      self.A_inv[action_v] = np.linalg.inv(self.A[action_v])
      self.b_vec[action_v] = np.dot(z.T, y)
      self.theta_hat[action_v] = np.dot(self.A_inv[action_v], self.b_vec[action_v])

  @property
  def lambda_prior(self):
    return self._lambda_prior
