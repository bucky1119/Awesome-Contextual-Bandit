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
    # 这里使用可调的探索系数 alpha
    # 实践中通常作为超参数调节，而不是严格使用论文中的 beta_t
    self._lambda_prior = hparams.get("lambda_prior", 1.0)  # 梯度 Gram 矩阵正则化参数
    self.update_freq_nn = hparams.get("training_freq_network",   hparams.get("training_freq", 50))  # NN 训练频率
    self.num_epochs = hparams.get("training_epochs", 100)  # NN 训练轮数
    self.initial_pulls = hparams.get("initial_pulls", 2)  # 初始轮询次数
    # 隐藏层宽度 m，用于 NTK 归一化（论文要求梯度除以 sqrt(m)）
    self.hidden_size = hparams.get("layer_sizes", [100])[0]
    self.t = 0  # 全局时间步

    # 为了避免 one-hot 特征尺度过大影响梯度
    # 对 action one-hot 做缩放，这样可以让 context 特征与 action 特征保持更接近的尺度
    self.arm_scale = getattr(hparams, "arm_scale", 0.1)

    # arm-augmented 特征维度：上下文 + one-hot arm 编码（与 File1 理论一致）
    # 每个 arm 的输入 = [context | one_hot(arm)]，使得 arm 间特征独立
    self.aug_dim = self.context_dim + self.num_actions

    # 数据集存储 augmented 上下文（维度 aug_dim）和标量奖励（单输出）
    self.data_h = ContextualDataset(
        np.empty((0, self.aug_dim)),
        np.empty((0, 1))
    )

    # 建立单输出神经网络模型：f([context; one_hot(a)]; θ) → 标量
    # 修改 hparams 以匹配 augmented 输入维度和标量输出
    aug_hparams = dict(hparams)
    aug_hparams["context_dim"] = self.aug_dim  # 输入：上下文 + arm one-hot
    aug_hparams["num_actions"] = 1             # 输出：标量（单输出网络）
    self.bnn = NeuralBanditModel(aug_hparams, name=f"{name}-bnn")

    # 计算网络总参数量 p（用于梯度向量维度）
    # 用于构建参数空间的协方差矩阵
    self.total_param_dim = sum(param.numel() for param in self.bnn.parameters())

    # 旧实现：每个 action 一个对角近似矩阵（理论上不准确）
    # self.Z_inv_diag = [
    #     (1.0 / self._lambda_prior) * np.ones(self.p)
    #     for _ in range(self.num_actions)
    # ]

    # 新实现：使用一个全局共享的完整逆协方差矩阵
    # 这样更接近 NeuralUCB 论文中的参数空间置信椭球 g^T Z^{-1} g
    # NeuralUCB 的不确定性是在共享网络参数空间上构建的，而不是 per-arm 的
    self.Z_inv = (1.0 / self._lambda_prior) * np.eye(self.total_param_dim, dtype=np.float64)

  def _augment_context(self, context, arm_idx):
    """Append one-hot arm encoding to context (arm-specific feature, as in File1).

    Args:
      context: Raw context vector of shape (context_dim,).
      arm_idx: Arm index to encode.

    Returns:
      Augmented vector of shape (aug_dim,) = [context | one_hot(arm_idx)].
    """
    one_hot = np.zeros(self.num_actions, dtype=np.float32)
    # 对 one-hot action 进行缩放，避免特征尺度过大影响梯度
    # 这样可以让 context 特征与 action 特征保持更接近的尺度
    one_hot[arm_idx] = self.arm_scale
    return np.concatenate([context.astype(np.float32), one_hot])

  def _get_gradient(self, context, action_idx):
    """Compute the gradient of the scalar network output w.r.t. parameters.

    Uses arm-specific augmented context [context | one_hot(action_idx)] as input
    to the single-output network, matching the theoretical setup of Zhou et al. (2020):
      g_a(x; θ) = ∇_θ f([x; e_a]; θ)  where e_a is the one-hot arm vector.

    Each arm's gradient is computed from an independent input, so Z_a matrices
    are semantically independent (no cross-arm contamination).

    Args:
      context: Raw context vector of shape (context_dim,).
      action_idx: Index of the action to compute gradient for.

    Returns:
      grad: Flattened, NTK-normalized gradient vector of shape (p,), dtype float64.
    """
    # 切换到 eval 模式（关闭 dropout），保证 UCB 计算确定性，结束后恢复原状态
    was_training = self.bnn.training
    self.bnn.eval()

    aug_ctx = self._augment_context(context, action_idx)
    ctx_tensor = torch.tensor(aug_ctx, dtype=torch.float32).unsqueeze(0)

    # 前向传播：单输出网络，output shape = (1, 1)
    self.bnn.zero_grad()
    output = self.bnn(ctx_tensor)
    output[0, 0].backward()  # 单标量输出，直接对唯一输出求梯度

    # 拼接所有参数的梯度为一个向量
    grads = []
    for param in self.bnn.parameters():
      if param.grad is not None:
        grads.append(param.grad.detach().cpu().numpy().flatten())
      else:
        grads.append(np.zeros(param.numel()))

    if was_training:
      self.bnn.train()

    # 这里使用 NeuralUCB 中的梯度特征
    # g(x,a) = ∇θ f(x,a)
    # 用于构建局部线性化的置信区间
    # NTK 归一化（论文理论保证的必要条件）：g / sqrt(m)，m 为隐藏层宽度
    # 同时统一为 float64，与 Z_inv 矩阵精度对齐
    grad = np.concatenate(grads).astype(np.float64) / np.sqrt(self.hidden_size)
    return grad

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

    # 对每个 arm 分别构造 augmented 输入，用单输出网络独立预测奖励
    ucb_values = np.zeros(self.num_actions)
    self.bnn.eval()

    for a in range(self.num_actions):
      aug_ctx = self._augment_context(context, a)
      aug_tensor = torch.tensor(aug_ctx, dtype=torch.float32).unsqueeze(0)

      # 单输出网络预测该 arm 的奖励（eval 模式，关闭 dropout）
      with torch.no_grad():
        pred = self.bnn(aug_tensor).item()

      # 计算该 arm 的梯度及置信宽度
      g = self._get_gradient(context, a)

      # 旧实现：使用对角近似的 g^2 * Z_inv_diag
      # confidence = self.alpha * np.sqrt(np.sum(g ** 2 * self.Z_inv_diag[a]))

      # 新实现：使用完整二次型 g^T Z^{-1} g
      # 这更接近 NeuralUCB 原论文中的置信区间表达
      # 使用全局共享的 Z_inv 矩阵来计算所有 arm 的不确定性
      quad = float(g @ self.Z_inv @ g)

      # 数值稳定性保护
      quad = max(quad, 0.0)

      confidence = self.alpha * np.sqrt(quad)
      ucb_values[a] = pred + confidence

    return int(np.argmax(ucb_values))

  def update(self, context, action, reward):
    """Updates the model and the gradient gram matrix.

    Args:
      context: Last observed context.
      action: Last observed action.
      reward: Last observed reward.
    """

    self.t += 1
    # 存储 augmented 上下文（arm-specific），action 固定为 0（单输出网络）
    aug_ctx = self._augment_context(context, action)
    self.data_h.add(aug_ctx, 0, reward)

    # 计算当前 arm 的梯度并更新 Z^{-1}
    g = self._get_gradient(context, action)

    # 旧实现：对角近似的 Sherman-Morrison 更新
    # g_sq = g ** 2
    # z_inv = self.Z_inv_diag[action]
    # denom = 1.0 + g_sq * z_inv
    # self.Z_inv_diag[action] = z_inv - (z_inv * g_sq * z_inv) / denom

    # 新实现：完整矩阵的 Sherman-Morrison rank-1 更新
    # 用于在线更新参数空间的逆协方差矩阵
    # 这样可以保持 NeuralUCB 理论中的置信椭球结构
    zg = self.Z_inv @ g
    denom = 1.0 + float(g @ zg)

    # 数值稳定保护
    if denom > 0:
      self.Z_inv = self.Z_inv - np.outer(zg, zg) / denom

    # 定期重训神经网络
    if self.t % self.update_freq_nn == 0:
      self.bnn.train_model(self.data_h, self.num_epochs)

  @property
  def lambda_prior(self):
    return self._lambda_prior
