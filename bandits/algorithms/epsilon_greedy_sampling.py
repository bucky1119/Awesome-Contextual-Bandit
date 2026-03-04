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

"""Epsilon-greedy algorithm for contextual bandits."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from bandits.core.bandit_algorithm import BanditAlgorithm


class EpsilonGreedySampling(BanditAlgorithm):
  """Epsilon-greedy baseline algorithm.

  With probability epsilon, selects a random action (exploration).
  Otherwise, selects the action with the highest empirical mean reward (exploitation).
  Supports an optional epsilon-decay schedule: epsilon_t = epsilon / (1 + decay * t).
  """

  def __init__(self, name, hparams):
    """Creates an EpsilonGreedySampling object.

    Args:
      name: Name of the algorithm.
      hparams: Hyper-parameters of the algorithm.
        Required attributes:
          - num_actions: Number of arms/actions.
        Optional attributes:
          - epsilon: Exploration probability (default: 0.1).
          - epsilon_decay: Decay rate for epsilon (default: 0.0, no decay).
    """

    self.name = name
    self.hparams = hparams
    self.num_actions = hparams.num_actions
    self.epsilon = getattr(hparams, 'epsilon', 0.1)  # 探索概率
    self.epsilon_decay = getattr(hparams, 'epsilon_decay', 0.0)  # epsilon 衰减率

    # 每个动作的累积奖励和和被选择次数
    self.reward_sum = np.zeros(self.num_actions)  # 每个动作的奖励和
    self.action_counts = np.zeros(self.num_actions)  # 每个动作被拉取的次数
    self.t = 0  # 全局时间步

  def _current_epsilon(self):
    """Compute current epsilon with optional decay.

    Returns:
      Current epsilon value.
    """
    return self.epsilon / (1.0 + self.epsilon_decay * self.t)

  def action(self, context):
    """Selects action using epsilon-greedy strategy (context is ignored).

    Args:
      context: Context for which the action need to be chosen (ignored).

    Returns:
      action: Selected action index.
    """

    # 初始阶段：轮流选择每个动作至少一次
    if self.t < self.num_actions:
      return self.t

    current_eps = self._current_epsilon()

    if np.random.random() < current_eps:
      # 探索：随机选择动作
      return np.random.choice(self.num_actions)
    else:
      # 利用：选择经验均值最高的动作
      mu_hat = self.reward_sum / np.maximum(self.action_counts, 1)
      return int(np.argmax(mu_hat))

  def update(self, context, action, reward):
    """Updates the empirical reward statistics.

    Args:
      context: Last observed context (ignored).
      action: Last observed action.
      reward: Last observed reward.
    """

    self.t += 1
    self.action_counts[action] += 1
    self.reward_sum[action] += reward
