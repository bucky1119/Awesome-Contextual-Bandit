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

"""UCB1 algorithm for contextual bandits (context-free upper confidence bound baseline)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from bandits.core.bandit_algorithm import BanditAlgorithm


class UCB1Sampling(BanditAlgorithm):
  """UCB1 (Upper Confidence Bound) baseline algorithm.

  This is a context-free bandit algorithm that maintains empirical mean rewards
  and selects the action with the highest upper confidence bound:
      a_t = argmax_a [ mu_hat(a) + alpha * sqrt(log(t) / N(a)) ]

  Reference:
      Auer, Cesa-Bianchi, Fischer (2002).
      Finite-time Analysis of the Multiarmed Bandit Problem.
  """

  def __init__(self, name, hparams):
    """Creates a UCB1Sampling object.

    Args:
      name: Name of the algorithm.
      hparams: Hyper-parameters of the algorithm.
        Required attributes:
          - num_actions: Number of arms/actions.
        Optional attributes:
          - alpha: Exploration coefficient (default: 1.0).
    """

    self.name = name
    self.hparams = hparams
    self.alpha = getattr(hparams, 'alpha', 1.0)  # 探索系数
    self.num_actions = hparams.num_actions

    # 每个动作的累积奖励和和被选择次数
    self.reward_sum = np.zeros(self.num_actions)  # 每个动作的奖励和
    self.action_counts = np.zeros(self.num_actions)  # 每个动作被拉取的次数
    self.t = 0  # 全局时间步

  def action(self, context):
    """Selects action with the highest UCB1 index (context is ignored).

    Args:
      context: Context for which the action need to be chosen (ignored by UCB1).

    Returns:
      action: Selected action index.
    """

    # 初始阶段：轮流选择每个动作至少一次
    if self.t < self.num_actions:
      return self.t

    # 计算每个动作的 UCB 指标
    mu_hat = self.reward_sum / self.action_counts  # 经验均值
    exploration_bonus = self.alpha * np.sqrt(np.log(self.t) / self.action_counts)  # 探索奖励
    ucb_values = mu_hat + exploration_bonus  # UCB 值

    return int(np.argmax(ucb_values))

  def update(self, context, action, reward):
    """Updates the empirical reward statistics.

    Args:
      context: Last observed context (ignored by UCB1).
      action: Last observed action.
      reward: Last observed reward.
    """

    self.t += 1
    self.action_counts[action] += 1
    self.reward_sum[action] += reward
