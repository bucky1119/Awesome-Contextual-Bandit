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

"""LinUCB algorithm for contextual bandits with disjoint linear models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from bandits.core.bandit_algorithm import BanditAlgorithm


class LinUCBSampling(BanditAlgorithm):
  """LinUCB (Linear Upper Confidence Bound) with disjoint linear models.

  For each action a, maintains a ridge regression model:
      reward = context^T theta_a + noise
  and selects the action with the highest UCB:
      a_t = argmax_a [ context^T theta_hat_a + alpha * sqrt(context^T A_a^{-1} context) ]

  Reference:
      Li, Chu, Langford, Schapire (2010).
      A Contextual-Bandit Approach to Personalized News Article Recommendation.
      WWW 2010.
  """

  def __init__(self, name, hparams):
    """Initialize LinUCB with per-arm ridge regression.

    Args:
      name: Name of the algorithm.
      hparams: Hyper-parameters of the algorithm.
        Required attributes:
          - num_actions: Number of arms/actions.
          - context_dim: Dimension of the context vectors.
        Optional attributes:
          - alpha: Exploration coefficient (default: 1.0).
          - lambda_prior: Ridge regularization parameter (default: 1.0).
    """

    self.name = name
    self.hparams = hparams
    self.num_actions = hparams.num_actions
    self.context_dim = hparams.context_dim
    self.alpha = getattr(hparams, 'alpha', 1.0)  # 探索系数
    self._lambda_prior = getattr(hparams, 'lambda_prior', 1.0)  # 岭回归正则化参数    # 初始轮询次数：保证每个 arm 至少被测试 initial_pulls 次。
    # 无此阶段时 t=0 所有 UCB 值相等，np.argmax 始终返回 arm 0，
    # 若 arm 0 早期奖励估计较高则其余 arm 长期得不到探索，有限轮次遗憾上界退化。
    self.initial_pulls = getattr(hparams, 'initial_pulls', 2)
    # 维度包含截距项
    self.d = self.context_dim + 1

    # 为每个动作初始化矩阵 A_a = lambda * I 和向量 b_a = 0
    self.A = [self._lambda_prior * np.eye(self.d) for _ in range(self.num_actions)]  # 精度矩阵
    self.A_inv = [(1.0 / self._lambda_prior) * np.eye(self.d) for _ in range(self.num_actions)]  # 精度矩阵的逆
    self.b = [np.zeros(self.d) for _ in range(self.num_actions)]  # 奖励加权上下文累加向量
    self.theta_hat = [np.zeros(self.d) for _ in range(self.num_actions)]  # 参数估计

    self.t = 0  # 全局时间步

  def _add_intercept(self, context):
    """Append a 1 to the context vector as intercept term.

    Args:
      context: Context vector of shape (context_dim,).

    Returns:
      Extended context vector of shape (context_dim + 1,).
    """
    return np.append(context, 1.0)

  def action(self, context):
    """Selects the action with the highest LinUCB index.

    Args:
      context: Context for which the action needs to be chosen.

    Returns:
      action: Selected action index.
    """

    # 初始阶段：轮流选择每个 arm，确保每个 arm 至少被观测 initial_pulls 次
    if self.t < self.num_actions * self.initial_pulls:
      return self.t % self.num_actions

    x = self._add_intercept(context)  # 扩展上下文向量，加入截距项

    ucb_values = np.zeros(self.num_actions)
    for a in range(self.num_actions):
      # 预测均值: x^T theta_hat_a
      pred = np.dot(self.theta_hat[a], x)
      # 置信宽度: alpha * sqrt(x^T A_a^{-1} x)
      confidence = self.alpha * np.sqrt(np.dot(x, np.dot(self.A_inv[a], x)))
      ucb_values[a] = pred + confidence

    return int(np.argmax(ucb_values))

  def update(self, context, action, reward):
    """Updates the linear model for the chosen action.

    Args:
      context: Last observed context.
      action: Last observed action.
      reward: Last observed reward.
    """

    self.t += 1
    x = self._add_intercept(context)

    # 使用 Sherman-Morrison 公式增量更新 A_inv
    # A_new = A_old + x x^T
    # A_inv_new = A_inv_old - (A_inv_old x x^T A_inv_old) / (1 + x^T A_inv_old x)
    A_inv_x = np.dot(self.A_inv[action], x)
    denom = 1.0 + np.dot(x, A_inv_x)
    self.A_inv[action] -= np.outer(A_inv_x, A_inv_x) / denom

    self.A[action] += np.outer(x, x)
    self.b[action] += reward * x
    self.theta_hat[action] = np.dot(self.A_inv[action], self.b[action])

  @property
  def lambda_prior(self):
    return self._lambda_prior
