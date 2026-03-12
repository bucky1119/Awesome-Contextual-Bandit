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

"""Thompson Sampling with linear posterior over a learnt deep representation."""

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
    self.name = name #算法名称，默认为neural_linear
    self.hparams = hparams #超参数字典
    self.latent_dim = self.hparams["layer_sizes"][-1] #最后一层的维度，作为线性回归的输入维度
    self._lambda_prior = self.hparams["lambda_prior"] #先验协方差的倒数
    self.mu = [np.zeros(self.latent_dim) for _ in range(self.hparams["num_actions"])] #每个动作的线性回归参数的均值初始化为0向量
    self.cov = [(1.0 / self._lambda_prior) * np.eye(self.latent_dim) for _ in range(self.hparams["num_actions"])] #协方差矩阵初始化为单位矩阵乘以先验倒数
    self.precision = [self._lambda_prior * np.eye(self.latent_dim) for _ in range(self.hparams["num_actions"])] #精度矩阵初始化为单位矩阵乘以先验
    self._a0 = self.hparams["a0"] #先验参数a0
    self._b0 = self.hparams["b0"] #先验参数b0
    self.a = [self._a0 for _ in range(self.hparams["num_actions"])] #后验参数a初始化为a0列表
    self.b = [self._b0 for _ in range(self.hparams["num_actions"])] #后验参数b初始化为b0列表
    self.update_freq_lr = self.hparams["training_freq"] #线性回归更新频率
    self.update_freq_nn = self.hparams["training_freq_network"] #神经网络训练频率
    self.t = 0 #时间步计数器
    self.num_epochs = self.hparams["training_epochs"] #神经网络训练轮数
    self.data_h = ContextualDataset(np.empty((0, self.hparams["context_dim"])), np.empty((0, self.hparams["num_actions"]))) #原始上下文数据集
    self.latent_h = ContextualDataset(np.empty((0, self.latent_dim)), np.empty((0, self.hparams["num_actions"]))) #潜在表示数据集
    self.bnn = NeuralBanditModel(self.hparams, name=f"{name}-bnn") #神经网络模型实例

  def action(self, context):
    """Samples beta's from posterior, and chooses best action accordingly."""

    # Round robin until each action has been selected "initial_pulls" times
    # 轮流选择动作，直到每个动作被选择了"initial_pulls"次，也就是初始阶段随便来一个动作进行探索
    if self.t < self.hparams["num_actions"] * self.hparams["initial_pulls"]: #初始拉取阶段，选择动作按顺序循环
      return self.t % self.hparams["num_actions"] #返回当前时间步对应的动作编号

    # Sample sigma2, and beta conditional on sigma2
    # 为每个动作分别采样一个方差参数sigma2
    sigma2_s = [
        self.b[i] * invgamma.rvs(self.a[i]) #通过逆伽马分布采样sigma2
        for i in range(self.hparams["num_actions"]) 
    ]

    try:
      # 为每个动作采样beta参数
      beta_s = [
        np.random.multivariate_normal(
          self.mu[i],
          self._stable_covariance(sigma2_s[i] * self.cov[i]),
          check_valid='ignore',
        ) #多元高斯分布采样beta
          for i in range(self.hparams["num_actions"])
      ]
    except np.linalg.LinAlgError as e:
      # Sampling could fail if covariance is not positive definite
      print(f'Exception when sampling for {self.name}. Details: {e}')
      d = self.latent_dim
      beta_s = [
          np.random.multivariate_normal(np.zeros((d)), np.eye(d))
          for _ in range(self.hparams["num_actions"])
      ]

    # Compute last-layer representation for the current context
    #计算当前上下文的最后一层表示
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0) #将上下文转换为张量并添加批次维度，得到可传入模型的context_tensor
    # 通过unsqueeze(0)在最左侧新增一个长度为 1 的批量维度，满足 PyTorch 模型对批量输入的要求。
    # 1. 进入不计算梯度的上下文
    #这段代码是模型推理 / 特征提取阶段（不是训练阶段），不需要计算梯度（梯度仅用于训练时的反向传播更新参数）。
    with torch.no_grad():
      # 2. 模型前向传播（仅运行network的前n-1层），得到张量输出
      # 3. 将PyTorch张量转换为NumPy数组
      # 4. 移除第0维（长度为1的批量维度），得到最终的上下文表示向量
      z_context = self.bnn.network[:-1](context_tensor).numpy().squeeze(0)  # Exclude last layer

    # Apply Thompson Sampling to last-layer representation
    # 对最后一层表示应用Thompson采样
    vals = [
        np.dot(beta_s[i], z_context) for i in range(self.hparams["num_actions"]) #例如神经网络维度为100，beta_s[i]也是100维，点积得到动作i的值，z_context是context最后一层网络的表征，也是100维
    ]
    return int(np.argmax(vals)) #选择值最大的动作返回

  def _stable_covariance(self, cov, eps=1e-8):
    """Numerically stabilize covariance to be symmetric PSD."""
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)

    # Project to PSD by clipping tiny negative eigenvalues caused by precision errors.
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, eps)
    cov_psd = (v * w) @ v.T
    cov_psd = 0.5 * (cov_psd + cov_psd.T)
    return cov_psd

  def update(self, context, action, reward):
    """Updates the posterior using linear bayesian regression formula."""

    self.t += 1 #时间步加1
    self.data_h.add(context, action, reward) #将新的（context, action, reward）添加到原始数据集中
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0) #将上下文转换为张量并添加批次维度
    with torch.no_grad(): #不计算梯度
      z_context = self.bnn.network[:-1](context_tensor).numpy().squeeze(0) #计算上下文的最后一层表示
    self.latent_h.add(z_context, action, reward) #将新的（latent context, action, reward）添加到潜在表示数据集中

    # Retrain the network on the original data (data_h)
    if self.t % self.update_freq_nn == 0:
      self.bnn.train_model(self.data_h, self.num_epochs) #使用原始数据集训练神经网络

      # Update the latent representation of every datapoint collected so far
      all_contexts = self.data_h.contexts.numpy() #获取所有原始上下文数据
      with torch.no_grad():
        new_z = self.bnn.network[:-1](torch.tensor(all_contexts, dtype=torch.float32)).numpy() #计算所有上下文的最后一层表示
      self.latent_h.contexts = torch.tensor(new_z, dtype=torch.float32) #更新潜在表示数据集的上下文为新的表示

    # Update the Bayesian Linear Regression
    if self.t % self.update_freq_lr == 0:
      # 更新贝叶斯线性回归模型
      actions_to_update = self.latent_h.actions[:-self.update_freq_lr] if self.update_freq_lr < len(self.latent_h.actions) else self.latent_h.actions
      for action_v in np.unique(actions_to_update):
        z, y = self.latent_h.get_batch_for_action(action_v)
        s = np.dot(z.T, z)
        precision_a = s + self.lambda_prior * np.eye(self.latent_dim)
        cov_a = np.linalg.inv(precision_a)
        mu_a = np.dot(cov_a, np.dot(z.T, y))
        a_post = self.a0 + z.shape[0] / 2.0
        b_upd = 0.5 * np.dot(y.T, y) - 0.5 * np.dot(mu_a.T, np.dot(precision_a, mu_a))
        b_post = self.b0 + b_upd
        self.mu[action_v] = mu_a
        self.cov[action_v] = cov_a
        self.precision[action_v] = precision_a
        self.a[action_v] = a_post
        self.b[action_v] = b_post

  @property
  def a0(self):
    return self._a0

  @property
  def b0(self):
    return self._b0

  @property
  def lambda_prior(self):
    return self._lambda_prior