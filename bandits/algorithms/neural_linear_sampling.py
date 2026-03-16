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

"""Thompson Sampling with linear posterior over a learnt deep representation.

数值更稳版本：
1. 对协方差矩阵做更稳的对称化与 PSD 修正；
2. 重建 BLR 后验时避免直接裸用矩阵求逆，优先用 solve；
3. 对 b_post、precision、cov 做下界和有限值检查；
4. 采样时对异常协方差进行多级回退，尽量避免 SVD did not converge。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import invgamma
import torch

from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset import ContextualDataset
from bandits.algorithms.neural_bandit_model import NeuralBanditModel


class NeuralLinearPosteriorSampling(BanditAlgorithm):
  """Full Bayesian linear regression on the last layer of a deep neural net (PyTorch version)."""

  def __init__(self, hparams, name="neural_linear"):
    self.name = name  # 算法名称，默认为 neural_linear
    self.hparams = hparams  # 超参数字典
    self.latent_dim = self.hparams["layer_sizes"][-1]  # 最后一层维度，作为线性回归的输入维度
    self._lambda_prior = self.hparams["lambda_prior"]  # 先验精度系数

    # 数值稳定相关超参数。
    self._jitter = float(self.hparams.get("jitter", 1e-6))
    self._min_b = float(self.hparams.get("min_b", 1e-8))
    self._max_sigma2 = float(self.hparams.get("max_sigma2", 1e6))

    # 每个动作一套 BLR 后验参数。
    self.mu = [np.zeros(self.latent_dim, dtype=np.float64)
               for _ in range(self.hparams["num_actions"])]
    self.cov = [(1.0 / self._lambda_prior) * np.eye(self.latent_dim, dtype=np.float64)
                for _ in range(self.hparams["num_actions"])]
    self.precision = [self._lambda_prior * np.eye(self.latent_dim, dtype=np.float64)
                      for _ in range(self.hparams["num_actions"])]

    self._a0 = self.hparams["a0"]
    self._b0 = self.hparams["b0"]
    self.a = [self._a0 for _ in range(self.hparams["num_actions"])]
    self.b = [self._b0 for _ in range(self.hparams["num_actions"])]

    self.update_freq_lr = self.hparams["training_freq"]
    self.update_freq_nn = self.hparams["training_freq_network"]
    self.t = 0
    self.num_epochs = self.hparams["training_epochs"]

    self.data_h = ContextualDataset(
        np.empty((0, self.hparams["context_dim"])),
        np.empty((0, self.hparams["num_actions"]))
    )
    self.latent_h = ContextualDataset(
        np.empty((0, self.latent_dim)),
        np.empty((0, self.hparams["num_actions"]))
    )
    self.bnn = NeuralBanditModel(self.hparams, name=f"{name}-bnn")

  def action(self, context):
    """从最后一层线性后验中采样 beta，并据此做 Thompson Sampling。"""

    # 初始阶段：确保每个动作都被选到 initial_pulls 次。
    if self.t < self.hparams["num_actions"] * self.hparams["initial_pulls"]:
      return self.t % self.hparams["num_actions"]

    # 为每个动作采样 sigma^2。
    sigma2_s = []
    for i in range(self.hparams["num_actions"]):
      # 数值保护：b 必须为正；同时对 sigma2 做截断，避免异常大尺度把协方差炸掉。
      b_i = float(max(self.b[i], self._min_b))
      try:
        sigma2_i = float(b_i * invgamma.rvs(self.a[i]))
      except Exception:
        sigma2_i = b_i
      if not np.isfinite(sigma2_i):
        sigma2_i = b_i
      sigma2_i = float(np.clip(sigma2_i, self._min_b, self._max_sigma2))
      sigma2_s.append(sigma2_i)

    beta_s = []
    for i in range(self.hparams["num_actions"]):
      # 对后验协方差做稳定化，尽量避免 multivariate_normal 内部 SVD 失败。
      cov_i = self._stable_covariance(sigma2_s[i] * self.cov[i])
      try:
        beta_i = np.random.multivariate_normal(
            mean=self.mu[i],
            cov=cov_i,
            check_valid='ignore',
        )
      except Exception as e:
        # 第一层回退：进一步增大 jitter 后再试一次。
        try:
          cov_i = self._stable_covariance(cov_i, eps=1e-6)
          beta_i = np.random.multivariate_normal(
              mean=self.mu[i],
              cov=cov_i,
              check_valid='ignore',
          )
        except Exception:
          # 第二层回退：使用对角近似协方差，至少保证能继续跑。
          print(f'Exception when sampling for {self.name}. Details: {e}')
          diag = np.maximum(np.diag(cov_i), self._min_b)
          beta_i = self.mu[i] + np.random.randn(self.latent_dim) * np.sqrt(diag)
      beta_s.append(beta_i)

    # 计算当前 context 的最后一层表示。
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
    self.bnn.eval()
    with torch.no_grad():
      z_context = self.bnn.network[:-1](context_tensor).detach().cpu().numpy().squeeze(0)

    # 若特征出现 NaN / Inf，则退化为零向量，避免后续点积传播异常值。
    if not np.isfinite(z_context).all():
      z_context = np.zeros(self.latent_dim, dtype=np.float64)
    else:
      z_context = np.asarray(z_context, dtype=np.float64)

    vals = [float(np.dot(beta_s[i], z_context)) for i in range(self.hparams["num_actions"])]
    return int(np.argmax(vals))

  def _stable_covariance(self, cov, eps=None):
    """将协方差矩阵稳定化为对称半正定矩阵。

    做法：
    1. 转成 float64；
    2. 强制对称化；
    3. 若存在 NaN / Inf，则回退到 eps * I；
    4. 用特征值分解裁掉过小/负的特征值；
    5. 再加一层对角 jitter，提高采样稳定性。
    """
    if eps is None:
      eps = self._jitter

    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)

    if not np.isfinite(cov).all():
      return eps * np.eye(self.latent_dim, dtype=np.float64)

    try:
      w, v = np.linalg.eigh(cov)
      w = np.maximum(w, eps)
      cov_psd = (v * w) @ v.T
      cov_psd = 0.5 * (cov_psd + cov_psd.T)
      cov_psd += eps * np.eye(cov_psd.shape[0], dtype=np.float64)
      return cov_psd
    except np.linalg.LinAlgError:
      # 特征分解失败时，直接退回对角矩阵，避免整个算法中断。
      diag = np.maximum(np.nan_to_num(np.diag(cov), nan=eps, posinf=eps, neginf=eps), eps)
      return np.diag(diag)

  def _rebuild_blr_params(self):
    """用当前 latent representations 从头重建全部 BLR 后验参数。

    只要 NN 重训过，特征空间就发生变化，因此必须重建 BLR 后验，
    保证 action() 中 Thompson Sampling 使用的后验与当前网络特征空间一致。
    """
    unique_actions = np.unique(self.latent_h.actions)
    for action_v in unique_actions:
      action_v = int(action_v)
      z, y = self.latent_h.get_batch_for_action(action_v)
      if len(z) == 0:
        continue

      # 统一转成 float64，并清除异常值。
      z = np.asarray(z, dtype=np.float64)
      y = np.asarray(y, dtype=np.float64).reshape(-1)

      # 若出现 NaN / Inf，仅保留有效样本，避免污染后验参数。
      valid_rows = np.isfinite(z).all(axis=1) & np.isfinite(y)
      z = z[valid_rows]
      y = y[valid_rows]
      if z.shape[0] == 0:
        continue

      # s = Z^T Z；加 jitter 到 precision 上，改善条件数。
      s = np.dot(z.T, z)
      precision_a = s + self._lambda_prior * np.eye(self.latent_dim, dtype=np.float64)
      precision_a += self._jitter * np.eye(self.latent_dim, dtype=np.float64)
      precision_a = 0.5 * (precision_a + precision_a.T)

      # 旧实现：直接 inv(precision_a)
      # cov_a = np.linalg.inv(precision_a)
      # mu_a = np.dot(cov_a, np.dot(z.T, y))

      # 新实现：优先使用 solve，数值上通常比显式求逆更稳。
      try:
        zy = np.dot(z.T, y)
        mu_a = np.linalg.solve(precision_a, zy)
        cov_a = np.linalg.solve(precision_a, np.eye(self.latent_dim, dtype=np.float64))
      except np.linalg.LinAlgError:
        # 若 solve 失败，则退回伪逆，避免单个动作导致整个重建失败。
        precision_a = precision_a + 1e-4 * np.eye(self.latent_dim, dtype=np.float64)
        cov_a = np.linalg.pinv(precision_a)
        mu_a = np.dot(cov_a, np.dot(z.T, y))

      cov_a = self._stable_covariance(cov_a)

      a_post = float(self._a0 + z.shape[0] / 2.0)

      # 旧实现：
      # b_upd = 0.5 * y^T y - 0.5 * mu^T precision mu
      # b_post = b0 + b_upd
      # 在浮点误差下，b_post 可能变成负数或极小值，导致后续 sigma^2 采样不稳定。
      quadratic = float(np.dot(mu_a.T, np.dot(precision_a, mu_a)))
      b_upd = 0.5 * float(np.dot(y.T, y)) - 0.5 * quadratic
      b_post = float(self._b0 + b_upd)
      b_post = float(max(b_post, self._min_b))

      # 最终写回前再做一次有限值检查；若失败，则跳过本次动作更新，保留旧后验。
      if not (np.isfinite(mu_a).all() and np.isfinite(cov_a).all() and np.isfinite(precision_a).all()):
        continue

      self.mu[action_v] = mu_a
      self.cov[action_v] = cov_a
      self.precision[action_v] = precision_a
      self.a[action_v] = a_post
      self.b[action_v] = b_post

  def update(self, context, action, reward):
    """更新原始数据、latent 表示，并按计划更新 BLR 后验。"""

    self.t += 1
    self.data_h.add(context, action, reward)

    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
    self.bnn.eval()
    with torch.no_grad():
      z_context = self.bnn.network[:-1](context_tensor).detach().cpu().numpy().squeeze(0)

    # 若当前表示异常，则退化为零向量，避免污染 latent_h。
    if not np.isfinite(z_context).all():
      z_context = np.zeros(self.latent_dim, dtype=np.float32)

    self.latent_h.add(z_context, action, reward)

    # 定期重训网络。
    if self.t % self.update_freq_nn == 0:
      self.bnn.train_model(self.data_h, self.num_epochs)

      # NN 重训后，重新计算所有历史 context 的 latent representation。
      self.bnn.eval()
      all_contexts = self.data_h.contexts.detach().cpu().numpy()
      with torch.no_grad():
        new_z = self.bnn.network[:-1](
            torch.tensor(all_contexts, dtype=torch.float32)
        ).detach().cpu().numpy()

      # 对异常表示做保护，避免后验重建时出现 NaN / Inf。
      new_z = np.asarray(new_z, dtype=np.float32)
      new_z = np.nan_to_num(new_z, nan=0.0, posinf=0.0, neginf=0.0)
      self.latent_h.contexts = torch.tensor(new_z, dtype=torch.float32)

      # NN 重训后必须立即重建 BLR 后验。
      self._rebuild_blr_params()

    # 若本轮未重训 NN，但到了 BLR 更新时间，也要重建一次后验。
    elif self.t % self.update_freq_lr == 0:
      self._rebuild_blr_params()

  @property
  def a0(self):
    return self._a0

  @property
  def b0(self):
    return self._b0

  @property
  def lambda_prior(self):
    return self._lambda_prior

# # Copyright 2018 The TensorFlow Authors All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================

# """Thompson Sampling with linear posterior over a learnt deep representation."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import numpy as np
# from scipy.stats import invgamma
# import torch

# from bandits.core.bandit_algorithm import BanditAlgorithm
# from bandits.core.contextual_dataset import ContextualDataset
# from bandits.algorithms.neural_bandit_model import NeuralBanditModel


# class NeuralLinearPosteriorSampling(BanditAlgorithm):
#   """Full Bayesian linear regression on the last layer of a deep neural net (PyTorch version)."""

#   def __init__(self, hparams, name="neural_linear"):
#     self.name = name #算法名称，默认为neural_linear
#     self.hparams = hparams #超参数字典
#     self.latent_dim = self.hparams["layer_sizes"][-1] #最后一层的维度，作为线性回归的输入维度
#     self._lambda_prior = self.hparams["lambda_prior"] #先验协方差的倒数
#     self.mu = [np.zeros(self.latent_dim) for _ in range(self.hparams["num_actions"])] #每个动作的线性回归参数的均值初始化为0向量
#     self.cov = [(1.0 / self._lambda_prior) * np.eye(self.latent_dim) for _ in range(self.hparams["num_actions"])] #协方差矩阵初始化为单位矩阵乘以先验倒数
#     self.precision = [self._lambda_prior * np.eye(self.latent_dim) for _ in range(self.hparams["num_actions"])] #精度矩阵初始化为单位矩阵乘以先验
#     self._a0 = self.hparams["a0"] #先验参数a0
#     self._b0 = self.hparams["b0"] #先验参数b0
#     self.a = [self._a0 for _ in range(self.hparams["num_actions"])] #后验参数a初始化为a0列表
#     self.b = [self._b0 for _ in range(self.hparams["num_actions"])] #后验参数b初始化为b0列表
#     self.update_freq_lr = self.hparams["training_freq"] #线性回归更新频率
#     self.update_freq_nn = self.hparams["training_freq_network"] #神经网络训练频率
#     self.t = 0 #时间步计数器
#     self.num_epochs = self.hparams["training_epochs"] #神经网络训练轮数
#     self.data_h = ContextualDataset(np.empty((0, self.hparams["context_dim"])), np.empty((0, self.hparams["num_actions"]))) #原始上下文数据集
#     self.latent_h = ContextualDataset(np.empty((0, self.latent_dim)), np.empty((0, self.hparams["num_actions"]))) #潜在表示数据集
#     self.bnn = NeuralBanditModel(self.hparams, name=f"{name}-bnn") #神经网络模型实例

#   def action(self, context):
#     """Samples beta's from posterior, and chooses best action accordingly."""

#     # Round robin until each action has been selected "initial_pulls" times
#     # 轮流选择动作，直到每个动作被选择了"initial_pulls"次，也就是初始阶段随便来一个动作进行探索
#     if self.t < self.hparams["num_actions"] * self.hparams["initial_pulls"]: #初始拉取阶段，选择动作按顺序循环
#       return self.t % self.hparams["num_actions"] #返回当前时间步对应的动作编号

#     # Sample sigma2, and beta conditional on sigma2
#     # 为每个动作分别采样一个方差参数sigma2
#     sigma2_s = [
#         self.b[i] * invgamma.rvs(self.a[i]) #通过逆伽马分布采样sigma2
#         for i in range(self.hparams["num_actions"]) 
#     ]

#     try:
#       # 为每个动作采样beta参数
#       beta_s = [
#         np.random.multivariate_normal(
#           self.mu[i],
#           self._stable_covariance(sigma2_s[i] * self.cov[i]),
#           check_valid='ignore',
#         ) #多元高斯分布采样beta
#           for i in range(self.hparams["num_actions"])
#       ]
#     except np.linalg.LinAlgError as e:
#       # Sampling could fail if covariance is not positive definite
#       print(f'Exception when sampling for {self.name}. Details: {e}')
#       d = self.latent_dim
#       beta_s = [
#           np.random.multivariate_normal(np.zeros((d)), np.eye(d))
#           for _ in range(self.hparams["num_actions"])
#       ]

#     # Compute last-layer representation for the current context
#     #计算当前上下文的最后一层表示
#     context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0) #将上下文转换为张量并添加批次维度，得到可传入模型的context_tensor
#     # 通过unsqueeze(0)在最左侧新增一个长度为 1 的批量维度，满足 PyTorch 模型对批量输入的要求。
#     # 1. 进入不计算梯度的上下文
#     #这段代码是模型推理 / 特征提取阶段（不是训练阶段），不需要计算梯度（梯度仅用于训练时的反向传播更新参数）。
#     self.bnn.eval()
#     with torch.no_grad():
#       # 2. 模型前向传播（仅运行network的前n-1层），得到张量输出
#       # 3. 将PyTorch张量转换为NumPy数组
#       # 4. 移除第0维（长度为1的批量维度），得到最终的上下文表示向量
#       z_context = self.bnn.network[:-1](context_tensor).numpy().squeeze(0)  # Exclude last layer

#     # Apply Thompson Sampling to last-layer representation
#     # 对最后一层表示应用Thompson采样
#     vals = [
#         np.dot(beta_s[i], z_context) for i in range(self.hparams["num_actions"]) #例如神经网络维度为100，beta_s[i]也是100维，点积得到动作i的值，z_context是context最后一层网络的表征，也是100维
#     ]
#     return int(np.argmax(vals)) #选择值最大的动作返回

#   def _stable_covariance(self, cov, eps=1e-8):
#     """Numerically stabilize covariance to be symmetric PSD."""
#     cov = np.asarray(cov, dtype=np.float64)
#     cov = 0.5 * (cov + cov.T)

#     # Project to PSD by clipping tiny negative eigenvalues caused by precision errors.
#     w, v = np.linalg.eigh(cov)
#     w = np.maximum(w, eps)
#     cov_psd = (v * w) @ v.T
#     cov_psd = 0.5 * (cov_psd + cov_psd.T)
#     return cov_psd

#   def _rebuild_blr_params(self):
#     """Rebuild all BLR posterior parameters from scratch using current latent representations.

#     Must be called after any update to latent_h.contexts (i.e., after NN retraining)
#     so that the posterior (μ, Σ, a, b) always lives in the same feature space as the
#     current network.  Calling this at other times is also fine — it is a full
#     batch recomputation, so the result is always exact.
#     """
#     for action_v in np.unique(self.latent_h.actions):
#       action_v = int(action_v)
#       z, y = self.latent_h.get_batch_for_action(action_v)
#       if len(z) == 0:
#         continue

#       s = np.dot(z.T, z)
#       precision_a = s + self._lambda_prior * np.eye(self.latent_dim)
#       cov_a = np.linalg.inv(precision_a)
#       mu_a = np.dot(cov_a, np.dot(z.T, y))
#       a_post = self._a0 + z.shape[0] / 2.0
#       b_upd = 0.5 * np.dot(y.T, y) - 0.5 * np.dot(mu_a.T, np.dot(precision_a, mu_a))
#       b_post = self._b0 + b_upd

#       self.mu[action_v] = mu_a
#       self.cov[action_v] = cov_a
#       self.precision[action_v] = precision_a
#       self.a[action_v] = a_post
#       self.b[action_v] = b_post

#   def update(self, context, action, reward):
#     """Updates the posterior using linear bayesian regression formula."""

#     self.t += 1 #时间步加1
#     self.data_h.add(context, action, reward) #将新的（context, action, reward）添加到原始数据集中
#     context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0) #将上下文转换为张量并添加批次维度
#     self.bnn.eval()
#     with torch.no_grad(): #不计算梯度
#       z_context = self.bnn.network[:-1](context_tensor).numpy().squeeze(0) #计算上下文的最后一层表示
#     self.latent_h.add(z_context, action, reward) #将新的（latent context, action, reward）添加到潜在表示数据集中

#     # Retrain the network on the original data (data_h)
#     if self.t % self.update_freq_nn == 0:
#       self.bnn.train_model(self.data_h, self.num_epochs) #使用原始数据集训练神经网络

#       # Update the latent representation of every datapoint collected so far
#       self.bnn.eval()
#       all_contexts = self.data_h.contexts.numpy() #获取所有原始上下文数据
#       with torch.no_grad():
#         new_z = self.bnn.network[:-1](torch.tensor(all_contexts, dtype=torch.float32)).numpy() #计算所有上下文的最后一层表示
#       self.latent_h.contexts = torch.tensor(new_z, dtype=torch.float32) #更新潜在表示数据集的上下文为新的表示

#       # 关键：NN 重训后特征空间变化，必须立即重建 BLR 后验，使 action() 中的
#       # Thompson 采样与当前网络特征空间保持一致。
#       # （若两个频率相同，此处重建后 elif 分支不再重复执行，无额外开销。）
#       self._rebuild_blr_params()

#     # Update the Bayesian Linear Regression at its own scheduled frequency
#     # using elif to avoid redundant rebuild on steps where NN was also retrained.
#     elif self.t % self.update_freq_lr == 0:
#       self._rebuild_blr_params()

#   @property
#   def a0(self):
#     return self._a0

#   @property
#   def b0(self):
#     return self._b0

#   @property
#   def lambda_prior(self):
#     return self._lambda_prior