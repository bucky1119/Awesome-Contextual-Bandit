# # Copyright 2026
# # Adaptive-Trust Residual LinUCB (自适应信任残差线性上置信界算法)
# #
# # ============================================================
# # 算法核心思想：
# #   总分数 = 信任权重 * 先验分数 + 残差均值 + 探索项
# #   score_a = trust_weight(a) * prior_score_a
# #             + residual_mean_a          (由残差模型预测)
# #             + alpha * residual_uncertainty_a  (UCB探索项)
# #
# # 其中残差目标为:
# #   y_res = reward - prior_score_a
# #   (即: 实际奖励 - 先验模型预测的分数)
# #
# # 信任权重机制:
# #   - 当先验预测误差大时，信任权重降低，减少对先验的依赖
# #   - 当先验预测误差小时，信任权重提高，更多依赖先验知识
# #   - 信任权重范围: [w_min, 1.0]
# #
# # 本实现适配 LinUCBSampling 的接口风格
# # ============================================================

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import numpy as np

# from bandits.core.bandit_algorithm import BanditAlgorithm


# class AdaptiveTrustResidualLinUCB(BanditAlgorithm):
#   """自适应信任残差LinUCB (Disjoint线性模型版本)

#   每个动作a维护一个岭回归模型来学习残差目标:
#       y_res = reward - prior_score_a  (实际奖励 - 先验分数)

#   动作选择公式:
#       score_a = w_a * prior_score_a   (信任加权后的先验分数)
#                 + x^T theta_hat_a     (残差模型的预测均值)
#                 + alpha * sqrt(x^T A_a^{-1} x)  (UCB探索上界)

#   其中 w_a 是根据先验预测误差自适应更新的信任权重。

#   --------------------------------------------------------------
#   支持的输入格式 (context参数):
#     1) 字典格式:
#          {
#            "context": np.ndarray(shape=(context_dim,)),    # 上下文向量
#            "prior_scores": np.ndarray(shape=(num_actions,)) # 先验分数(可选)
#          }
#     2) 元组/列表格式:
#          (context_vector, prior_scores_vector)
#     3) 纯numpy数组:
#          只提供context_vector，先验分数默认为全0

#   --------------------------------------------------------------
#   超参数说明:

#     必需参数:
#       - num_actions: 动作数量
#       - context_dim: 上下文维度

#     可选参数 (UCB/岭回归相关):
#       - alpha: 探索系数，控制UCB项的大小 (默认: 1.0)
#       - lambda_prior: 岭回归正则化系数 (默认: 1.0)
#       - initial_pulls: 初始强制探索次数，每个动作被拉取次数 (默认: 2)

#     可选参数 (信任机制相关):
#       - trust_alpha: 信任更新的EMA系数，控制误差追踪的平滑度 (默认: 0.05)
#       - lambda_trust: 信任衰减系数，误差越大权重下降越快 (默认: 5.0)
#       - trust_mode: 信任模式，"global"(全局) 或 "per_action"(每个动作独立) (默认: "global")
#       - w_min: 最小信任权重，防止完全忽略先验 (默认: 0.01)

#     可选参数 (先验分数处理):
#       - prior_clip_min: 先验分数下限 (默认: 0.0)
#       - prior_clip_max: 先验分数上限 (默认: 1.0)
#   """

#   def __init__(self, name, hparams):
#     self.name = name
#     self.hparams = hparams

#     self.num_actions = hparams.num_actions
#     self.context_dim = hparams.context_dim

#     # UCB / ridge params
#     self.alpha = getattr(hparams, 'alpha', 1.0)
#     self._lambda_prior = getattr(hparams, 'lambda_prior', 1.0)
#     self.initial_pulls = getattr(hparams, 'initial_pulls', 2)

#     # Trust params
#     self.trust_alpha = getattr(hparams, 'trust_alpha', 0.05)
#     self.lambda_trust = getattr(hparams, 'lambda_trust', 5.0)
#     self.trust_mode = getattr(hparams, 'trust_mode', 'global')
#     self.w_min = getattr(hparams, 'w_min', 0.01)

#     # Prior score clipping
#     self.prior_clip_min = getattr(hparams, 'prior_clip_min', 0.0)
#     self.prior_clip_max = getattr(hparams, 'prior_clip_max', 1.0)

#     # Dimension with intercept
#     self.d = self.context_dim + 1

#     # Disjoint LinUCB stats for residual learning
#     self.A = [self._lambda_prior * np.eye(self.d) for _ in range(self.num_actions)]
#     self.A_inv = [(1.0 / self._lambda_prior) * np.eye(self.d) for _ in range(self.num_actions)]
#     self.b = [np.zeros(self.d) for _ in range(self.num_actions)]
#     self.theta_hat = [np.zeros(self.d) for _ in range(self.num_actions)]

#     # Trust statistics
#     self.global_ema_error = 0.0
#     self.arm_ema_error = np.zeros(self.num_actions)

#     # Cache most recent action-time info for debugging/inspection
#     self.last_scores = None
#     self.last_priors = None
#     self.last_trust_weights = None

#     self.t = 0

#   # ---------------------------------------------------------------------------
#   # Helpers
#   # ---------------------------------------------------------------------------

#   def _add_intercept(self, context):
#     """添加截距项(偏置)

#     在上下文向量末尾添加常数1，用于岭回归的截距项。
#     输入: (context_dim,) -> 输出: (context_dim + 1,)
#     """
#     return np.append(context, 1.0)

#   def _extract_context_and_priors(self, context):
#     """从支持的输入格式中提取上下文向量和先验分数

#     支持三种输入格式:
#       - 字典: {"context": array, "prior_scores": array}
#       - 元组: (context_array, prior_scores_array)
#       - 纯数组: context_array (先验默认为0)

#     参数:
#       context: 上下文信息，支持上述三种格式

#     返回:
#       x_raw: 原始上下文向量，形状 (context_dim,)
#       prior_scores: 先验分数数组，形状 (num_actions,)
#     """
#     if isinstance(context, dict):
#       if 'context' not in context:
#         raise ValueError('If context is a dict, it must contain key "context".')
#       x_raw = np.asarray(context['context'], dtype=np.float64)

#       if 'prior_scores' in context and context['prior_scores'] is not None:
#         prior_scores = np.asarray(context['prior_scores'], dtype=np.float64)
#       else:
#         prior_scores = np.zeros(self.num_actions, dtype=np.float64)

#     elif isinstance(context, (tuple, list)) and len(context) == 2:
#       x_raw = np.asarray(context[0], dtype=np.float64)
#       prior_scores = np.asarray(context[1], dtype=np.float64)

#     else:
#       # Fallback: no prior information available
#       x_raw = np.asarray(context, dtype=np.float64)
#       prior_scores = np.zeros(self.num_actions, dtype=np.float64)

#     if x_raw.shape[0] != self.context_dim:
#       raise ValueError(
#           'Context dimension mismatch: got {}, expected {}'.format(
#               x_raw.shape[0], self.context_dim))

#     if prior_scores.shape[0] != self.num_actions:
#       raise ValueError(
#           'Prior score dimension mismatch: got {}, expected {}'.format(
#               prior_scores.shape[0], self.num_actions))

#     # Clip prior scores to a stable range
#     prior_scores = np.clip(prior_scores, self.prior_clip_min, self.prior_clip_max)

#     return x_raw, prior_scores

#   def _get_trust_weight(self, action):
#     """获取给定动作的信任权重

#     信任权重公式: w = 1 / (1 + lambda_trust * ema_error)

#     两种模式:
#       - global: 使用全局误差追踪，所有动作共享同一个信任权重
#       - per_action: 每个动作独立追踪误差，每个动作有不同信任权重

#     参数:
#       action: 动作索引

#     返回:
#       trust_weight: 信任权重，范围 [w_min, 1.0]
#     """
#     if self.trust_mode == 'global':
#       w = 1.0 / (1.0 + self.lambda_trust * self.global_ema_error)
#     elif self.trust_mode == 'per_action':
#       w = 1.0 / (1.0 + self.lambda_trust * self.arm_ema_error[action])
#     else:
#       raise ValueError('Unknown trust_mode: {}'.format(self.trust_mode))

#     return max(self.w_min, float(w))

#   def _update_trust(self, action, prior_score, reward):
#     """更新信任统计信息

#     使用指数移动平均(EMA)追踪先验预测误差:
#       ema_error = (1 - alpha) * ema_error + alpha * |reward - prior_score|

#     参数:
#       action: 被选择动作的索引
#       prior_score: 先验模型对该动作的预测分数
#       reward: 实际获得的奖励
#     """
#     err = abs(float(reward) - float(prior_score))

#     if self.trust_mode == 'global':
#       self.global_ema_error = (
#           (1.0 - self.trust_alpha) * self.global_ema_error
#           + self.trust_alpha * err
#       )
#     elif self.trust_mode == 'per_action':
#       self.arm_ema_error[action] = (
#           (1.0 - self.trust_alpha) * self.arm_ema_error[action]
#           + self.trust_alpha * err
#       )
#     else:
#       raise ValueError('Unknown trust_mode: {}'.format(self.trust_mode))

#   # ---------------------------------------------------------------------------
#   # Main API
#   # ---------------------------------------------------------------------------

#   def action(self, context):
#     """选择具有最高自适应信任残差LinUCB分数的动作

#     算法流程:
#       1. 初始探索阶段: 强制每个动作被选择initial_pulls次
#       2. 正常决策阶段:
#          - 对每个动作计算总分数
#          - 分数 = 信任权重 * 先验分数 + 残差预测 + UCB探索项
#          - 选择分数最高的动作

#     参数:
#       context: 上下文信息，支持字典/元组/数组格式

#     返回:
#       selected_action: 选择的动作索引 (int)
#     """
#     # 初始探索阶段: 强制每个动作被选择一定次数，确保冷启动
#     if self.t < self.num_actions * self.initial_pulls:
#       return self.t % self.num_actions

#     # 解析上下文和先验分数
#     x_raw, prior_scores = self._extract_context_and_priors(context)
#     x = self._add_intercept(x_raw)  # 添加截距项

#     scores = np.zeros(self.num_actions, dtype=np.float64)
#     trust_weights = np.zeros(self.num_actions, dtype=np.float64)

#     # 对每个动作计算UCB分数
#     for a in range(self.num_actions):
#       prior = float(prior_scores[a])  # 先验分数
#       trust_w = self._get_trust_weight(a)  # 获取信任权重

#       # 残差均值: x^T theta_hat_a (岭回归预测的残差)
#       residual_mean = np.dot(self.theta_hat[a], x)

#       # 残差不确定性 (UCB探索项)
#       uncertainty = self.alpha * np.sqrt(np.dot(x, np.dot(self.A_inv[a], x)))

#       # 总分数 = 信任加权先验 + 残差预测 + 探索项
#       scores[a] = trust_w * prior + residual_mean + uncertainty
#       trust_weights[a] = trust_w

#     # 缓存用于分析/调试
#     self.last_scores = scores.copy()
#     self.last_priors = prior_scores.copy()
#     self.last_trust_weights = trust_weights.copy()

#     # 返回分数最高的动作
#     return int(np.argmax(scores))

#   def update(self, context, action, reward):
#     """更新选中动作的模型参数

#     使用残差目标更新岭回归模型:
#       残差目标 = reward - prior_score
#       (即: 实际奖励 - 先验模型的预测分数)

#     参数:
#       context: 上下文信息
#       action: 被选中的动作索引
#       reward: 获得的奖励值 (0或1，用于分类任务)
#     """
#     self.t += 1

#     # 解析上下文
#     x_raw, prior_scores = self._extract_context_and_priors(context)
#     x = self._add_intercept(x_raw)  # 添加截距项

#     # 计算残差目标: 实际奖励与先验预测的差异
#     prior_score = float(prior_scores[action])
#     residual_reward = float(reward) - prior_score  # y_res = y - prior

#     # 使用Sherman-Morrison公式高效更新逆协方差矩阵
#     # 避免每次都重新求逆，提高计算效率
#     A_inv_x = np.dot(self.A_inv[action], x)
#     denom = 1.0 + np.dot(x, A_inv_x)
#     self.A_inv[action] -= np.outer(A_inv_x, A_inv_x) / denom

#     # 标准岭回归统计量更新 (对残差目标进行学习)
#     self.A[action] += np.outer(x, x)  # A = A + x*x^T
#     self.b[action] += residual_reward * x  # b = b + y_res * x
#     # 更新参数估计: theta = A^{-1} * b
#     self.theta_hat[action] = np.dot(self.A_inv[action], self.b[action])

#     # 更新信任统计: 追踪先验预测误差
#     self._update_trust(action, prior_score, reward)

#   # ---------------------------------------------------------------------------
#   # Properties / inspection
#   # ---------------------------------------------------------------------------

#   @property
#   def lambda_prior(self):
#     return self._lambda_prior

#   def get_state(self):
#     """Optional helper for logging/debugging."""
#     state = {
#         't': self.t,
#         'global_ema_error': self.global_ema_error,
#         'arm_ema_error': self.arm_ema_error.copy(),
#         'last_scores': None if self.last_scores is None else self.last_scores.copy(),
#         'last_priors': None if self.last_priors is None else self.last_priors.copy(),
#         'last_trust_weights': None if self.last_trust_weights is None else self.last_trust_weights.copy(),
#     }
#     return state


# Copyright 2026
# Static-Prior Residual LinUCB (静态先验残差线性上置信界算法)
#
# ============================================================
# 算法核心思想：
#   总分数 = 先验分数 + 残差均值 + 探索项
#   score_a = prior_score_a
#             + residual_mean_a
#             + alpha * residual_uncertainty_a
#
# 其中残差目标为:
#   y_res = reward - prior_score_a
#   (即: 实际奖励 - 先验模型预测的分数)
#
# 与 AdaptiveTrustResidualLinUCB 的区别：
#   - 本版本不使用 trust weight
#   - 动作选择时直接使用 prior_score_a
#   - 理论上更容易与“Residual LinUCB with Prior”对齐
#
# 本实现适配 LinUCBSampling 的接口风格
# ============================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from bandits.core.bandit_algorithm import BanditAlgorithm


# class StaticPriorResidualLinUCB(BanditAlgorithm):
class AdaptiveTrustResidualLinUCB(BanditAlgorithm):
  """静态先验残差 LinUCB (Disjoint 线性模型版本)

  每个动作 a 维护一个岭回归模型来学习残差目标:
      y_res = reward - prior_score_a

  动作选择公式:
      score_a = prior_score_a
                + x^T theta_hat_a
                + alpha * sqrt(x^T A_a^{-1} x)

  其中：
    - prior_score_a 由外部 LLM prior provider 提供
    - x^T theta_hat_a 是残差模型的预测均值
    - alpha * sqrt(...) 是 UCB 探索项

  --------------------------------------------------------------
  支持的输入格式 (context参数):
    1) 字典格式:
         {
           "context": np.ndarray(shape=(context_dim,)),      # 上下文向量
           "prior_scores": np.ndarray(shape=(num_actions,))  # 先验分数(可选)
         }
    2) 元组/列表格式:
         (context_vector, prior_scores_vector)
    3) 纯 numpy 数组:
         只提供 context_vector，先验分数默认为全0

  --------------------------------------------------------------
  超参数说明:

    必需参数:
      - num_actions: 动作数量
      - context_dim: 上下文维度

    可选参数:
      - alpha: 探索系数，控制UCB项大小 (默认: 1.0)
      - lambda_prior: 岭回归正则化系数 (默认: 1.0)
      - initial_pulls: 初始强制探索次数，每个动作被拉取次数 (默认: 2)

      - prior_clip_min: 先验分数下限 (默认: 0.0)
      - prior_clip_max: 先验分数上限 (默认: 1.0)
  """

  def __init__(self, name, hparams):
    self.name = name
    self.hparams = hparams

    self.num_actions = hparams.num_actions
    self.context_dim = hparams.context_dim

    # UCB / ridge params
    self.alpha = getattr(hparams, 'alpha', 1.0)
    self._lambda_prior = getattr(hparams, 'lambda_prior', 1.0)
    self.initial_pulls = getattr(hparams, 'initial_pulls', 2)

    # Prior score clipping
    self.prior_clip_min = getattr(hparams, 'prior_clip_min', 0.0)
    self.prior_clip_max = getattr(hparams, 'prior_clip_max', 1.0)

    # 上下文维度 + 截距项
    self.d = self.context_dim + 1

    # Disjoint LinUCB stats for residual learning
    self.A = [self._lambda_prior * np.eye(self.d) for _ in range(self.num_actions)]
    self.A_inv = [(1.0 / self._lambda_prior) * np.eye(self.d) for _ in range(self.num_actions)]
    self.b = [np.zeros(self.d) for _ in range(self.num_actions)]
    self.theta_hat = [np.zeros(self.d) for _ in range(self.num_actions)]

    # Cache recent info for debugging / inspection
    self.last_scores = None
    self.last_priors = None

    self.t = 0

  # ---------------------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------------------

  def _add_intercept(self, context):
    """添加截距项(偏置)

    在上下文向量末尾添加常数1，用于岭回归的截距项。
    输入: (context_dim,) -> 输出: (context_dim + 1,)
    """
    return np.append(context, 1.0)

  def _extract_context_and_priors(self, context):
    """从支持的输入格式中提取上下文向量和先验分数

    支持三种输入格式:
      - 字典: {"context": array, "prior_scores": array}
      - 元组: (context_array, prior_scores_array)
      - 纯数组: context_array (先验默认为0)

    参数:
      context: 上下文信息，支持上述三种格式

    返回:
      x_raw: 原始上下文向量，形状 (context_dim,)
      prior_scores: 先验分数数组，形状 (num_actions,)
    """
    if isinstance(context, dict):
      if 'context' not in context:
        raise ValueError('If context is a dict, it must contain key "context".')
      x_raw = np.asarray(context['context'], dtype=np.float64)

      if 'prior_scores' in context and context['prior_scores'] is not None:
        prior_scores = np.asarray(context['prior_scores'], dtype=np.float64)
      else:
        prior_scores = np.zeros(self.num_actions, dtype=np.float64)

    elif isinstance(context, (tuple, list)) and len(context) == 2:
      x_raw = np.asarray(context[0], dtype=np.float64)
      prior_scores = np.asarray(context[1], dtype=np.float64)

    else:
      # Fallback: no prior information available
      x_raw = np.asarray(context, dtype=np.float64)
      prior_scores = np.zeros(self.num_actions, dtype=np.float64)

    if x_raw.shape[0] != self.context_dim:
      raise ValueError(
          'Context dimension mismatch: got {}, expected {}'.format(
              x_raw.shape[0], self.context_dim))

    if prior_scores.shape[0] != self.num_actions:
      raise ValueError(
          'Prior score dimension mismatch: got {}, expected {}'.format(
              prior_scores.shape[0], self.num_actions))

    # 将 prior 限制到稳定区间
    prior_scores = np.clip(prior_scores, self.prior_clip_min, self.prior_clip_max)

    return x_raw, prior_scores

  # ---------------------------------------------------------------------------
  # Main API
  # ---------------------------------------------------------------------------

  def action(self, context):
    """选择具有最高静态先验残差 LinUCB 分数的动作

    算法流程:
      1. 初始探索阶段: 强制每个动作被选择 initial_pulls 次
      2. 正常决策阶段:
         - 对每个动作计算总分数
         - 分数 = 先验分数 + 残差预测 + UCB探索项
         - 选择分数最高的动作

    参数:
      context: 上下文信息，支持字典/元组/数组格式

    返回:
      selected_action: 选择的动作索引 (int)
    """
    # 初始探索阶段
    if self.t < self.num_actions * self.initial_pulls:
      return self.t % self.num_actions

    # 解析上下文和先验分数
    x_raw, prior_scores = self._extract_context_and_priors(context)
    x = self._add_intercept(x_raw)

    scores = np.zeros(self.num_actions, dtype=np.float64)

    # 对每个动作计算分数
    for a in range(self.num_actions):
      prior = float(prior_scores[a])

      # 残差均值: x^T theta_hat_a
      residual_mean = np.dot(self.theta_hat[a], x)

      # 残差不确定性 (UCB项)
      uncertainty = self.alpha * np.sqrt(np.dot(x, np.dot(self.A_inv[a], x)))

      # 总分数 = 静态先验 + 残差预测 + 探索项
      scores[a] = prior + residual_mean + uncertainty

    # 缓存用于分析/调试
    self.last_scores = scores.copy()
    self.last_priors = prior_scores.copy()

    return int(np.argmax(scores))

  def update(self, context, action, reward):
    """更新选中动作的模型参数

    使用残差目标更新岭回归模型:
      residual_target = reward - prior_score

    参数:
      context: 上下文信息
      action: 被选中的动作索引
      reward: 获得的奖励值
    """
    self.t += 1

    # 解析上下文
    x_raw, prior_scores = self._extract_context_and_priors(context)
    x = self._add_intercept(x_raw)

    # 残差目标: 实际奖励 - 先验预测
    prior_score = float(prior_scores[action])
    residual_reward = float(reward) - prior_score

    # 使用 Sherman-Morrison 公式更新逆矩阵
    A_inv_x = np.dot(self.A_inv[action], x)
    denom = 1.0 + np.dot(x, A_inv_x)
    self.A_inv[action] -= np.outer(A_inv_x, A_inv_x) / denom

    # 更新标准岭回归统计量
    self.A[action] += np.outer(x, x)
    self.b[action] += residual_reward * x
    self.theta_hat[action] = np.dot(self.A_inv[action], self.b[action])

  # ---------------------------------------------------------------------------
  # Properties / inspection
  # ---------------------------------------------------------------------------

  @property
  def lambda_prior(self):
    return self._lambda_prior

  def get_state(self):
    """用于日志记录 / 调试的状态接口。"""
    state = {
        't': self.t,
        'last_scores': None if self.last_scores is None else self.last_scores.copy(),
        'last_priors': None if self.last_priors is None else self.last_priors.copy(),
    }
    return state