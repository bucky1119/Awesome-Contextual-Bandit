import time
import csv
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

import abc
from tqdm import tqdm


def inv_sherman_morrison(u, A_inv):
    """计算矩阵的秩1更新逆矩阵 (Sherman-Morrison公式的逆)

    用于在在线学习场景中高效更新协方差矩阵的逆矩阵，避免每次重新计算矩阵逆。

    参数:
        u: 向量，表示梯度或特征
        A_inv: 矩阵A的逆矩阵
    返回:
        (A + u*u^T)^{-1} 的逆矩阵
    """
    Au = np.dot(A_inv, u)
    A_inv -= np.outer(Au, Au)/(1+np.dot(u.T, Au))
    return A_inv


class Model(nn.Module):
    """用于标量近似的全连接神经网络模板

    这是一个可配置的神经网络，用于近似上下文老虎机中的奖励函数。
    支持自定义隐藏层大小、层数、激活函数和dropout。
    """

    def __init__(self,
                 input_size=1,      # 输入特征维度
                 hidden_size=2,      # 隐藏层神经元数量
                 n_layers=1,         # 神经网络层数
                 activation='ReLU',  # 激活函数类型
                 p=0.0,              # dropout概率
                 ):
        super(Model, self).__init__()

        self.n_layers = n_layers

        # 构建网络层：单层或多层全连接网络
        if self.n_layers == 1:
            self.layers = [nn.Linear(input_size, 1)]  # 单层：直接映射到输出
        else:
            # 多层：输入层 -> 隐藏层 -> ... -> 输出层(1维)
            size = [input_size] + [hidden_size, ] * (self.n_layers - 1) + [1]
            self.layers = [nn.Linear(size[i], size[i + 1]) for i in range(self.n_layers)]
        self.layers = nn.ModuleList(self.layers)

        # dropout层，用于防止过拟合
        self.dropout = nn.Dropout(p=p)

        # 选择激活函数
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise Exception('{} not an available activation'.format(activation))

    def forward(self, x):
        """前向传播

        参数:
            x: 输入特征张量
        返回:
            预测的奖励值
        """
        for i in range(self.n_layers - 1):
            # 隐藏层：线性变换 -> 激活函数 -> Dropout
            x = self.dropout(self.activation(self.layers[i](x)))
        # 输出层：直接线性变换
        x = self.layers[-1](x)
        return x


class ContextualBandit():
    """上下文老虎机环境类

    定义了一个上下文老虎机问题，包含：
    - T 轮交互
    - n_arms 个手臂
    - n_features 维特征
    - 奖励函数
    """

    def __init__(self,
                 T,              # 总轮数
                 n_arms,         # 手臂数量
                 n_features,     # 特征维度
                 features,       # 上下文特征
                 rewards,        # 奖励数据
                 noise_std=1.0,  # 奖励噪声标准差
                 ):
        # 轮数
        self.T = T
        # 手臂数量
        self.n_arms = n_arms
        # 每个手臂的特征维度
        self.n_features = n_features

        # 高斯奖励噪声的标准差
        self.noise_std = noise_std

        # 初始化特征和奖励
        self.reset(features, rewards)

    @property
    def arms(self):
        """返回所有手臂的索引 [0, 1, ..., n_arms-1]
        """
        return range(self.n_arms)

    def reset(self, features, rewards):
        """重置环境数据

        参数:
            features: 上下文特征，形状为 (T, n_arms, n_features)
            rewards: 奖励，形状为 (T, n_arms)
        """
        self.features = features
        self.rewards = rewards
        # 计算每个时刻的最优奖励（用于计算regret）
        self.best_rewards_oracle = np.max(self.rewards, axis=1)
        # 记录每个时刻的最优手臂
        self.best_actions_oracle = np.argmax(self.rewards, axis=1)

    '''
    def reset(self):
        """Generate new features and new rewards.
        """
        self.reset_features()
        self.reset_rewards()

    def reset_features(self):
        """Generate normalized random N(0,1) features.
        """
        x = np.random.randn(self.T, self.n_arms, self.n_features)
        x /= np.repeat(np.linalg.norm(x, axis=-1, ord=2), self.n_features).reshape(self.T, self.n_arms, self.n_features)
        self.features = x

    def reset_rewards(self):
        """Generate rewards for each arm and each round,
        following the reward function h + Gaussian noise.
        """
        self.rewards = np.array(
            [
                self.h(self.features[t, k]) + self.noise_std * np.random.randn() \
                for t, k in itertools.product(range(self.T), range(self.n_arms))
            ]
        ).reshape(self.T, self.n_arms)

        # to be used only to compute regret, NOT by the algorithm itself
        self.best_rewards_oracle = np.max(self.rewards, axis=1)
        self.best_actions_oracle = np.argmax(self.rewards, axis=1)
    '''

class UCB(abc.ABC):
    """UCB (Upper Confidence Bound) 方法的基类

    实现了UCB算法的核心框架，包括：
    - 置信上界的计算和更新
    - 奖励预测
    - 探索与利用的平衡
    - 遗憾(regret)计算和记录

    UCB算法的核心思想是：在利用已知信息选择最优手臂的同时，
    通过置信上界来鼓励探索不确定的手臂。
    """

    def __init__(self,
                 bandit,                          # 老虎机环境对象
                 reg_factor=1.0,                  # L2正则化强度
                 confidence_scaling_factor=-1.0, # 置信度缩放因子
                 delta=0.1,                       # 置信概率参数
                 train_every=1,                   # 训练频率
                 throttle=int(1e2),               # 进度条更新频率
                 log_output_dir='./'             # 日志输出目录
                 ):
        # 老虎机对象，包含特征和生成的奖励
        self.bandit = bandit
        # L2正则化强度
        self.reg_factor = reg_factor
        # 置信上界概率 1-delta
        self.delta = delta
        # 置信上界的缩放因子（默认为奖励噪声标准差）
        if confidence_scaling_factor == -1.0:
            confidence_scaling_factor = bandit.noise_std
        self.confidence_scaling_factor = confidence_scaling_factor

        # 每隔几轮训练一次近似器
        self.train_every = train_every

        # 进度条更新频率
        self.throttle = throttle

        # 日志文件路径
        self.log_output_dir = log_output_dir

        self.iteration = 0
        self.reset()

    def reset_upper_confidence_bounds(self):
        """初始化置信上界及相关变量

        存储:
            exploration_bonus: 探索奖励数组
            mu_hat: 预测的奖励均值
            upper_confidence_bounds: 置信上界
        """
        self.exploration_bonus = np.empty((self.bandit.T, self.bandit.n_arms))
        self.mu_hat = np.empty((self.bandit.T, self.bandit.n_arms))
        self.upper_confidence_bounds = np.ones((self.bandit.T, self.bandit.n_arms))

    def reset_regrets(self):
        """初始化遗憾数组"""
        self.regrets = np.empty(self.bandit.T)

    def reset_actions(self):
        """初始化动作记录数组"""
        self.actions = np.empty(self.bandit.T).astype('int')

    def reset_A_inv(self):
        """初始化协方差矩阵逆的数组

        A_inv 用于计算UCB中的置信项，初始化为单位矩阵（正则化后）
        """
        self.A_inv = np.array(
            [
                np.eye(self.approximator_dim) / self.reg_factor for _ in self.bandit.arms
            ]
        )

    def reset_grad_approx(self):
        """初始化近似器关于参数的梯度"""
        self.grad_approx = np.zeros((self.bandit.n_arms, self.approximator_dim))

    def sample_action(self):
        """根据当前估计选择动作

        选择置信上界最大的手臂（UCB策略）
        """
        return np.argmax(self.upper_confidence_bounds[self.iteration]).astype('int')

    @abc.abstractmethod
    def reset(self):
        """初始化感兴趣的变量

        需要在子类中实现
        """
        pass

    @property
    @abc.abstractmethod
    def approximator_dim(self):
        """近似器使用的参数数量"""
        pass

    @property
    @abc.abstractmethod
    def confidence_multiplier(self):
        """置信探索奖励的乘数

        需要在子类中实现
        """
        pass

    @abc.abstractmethod
    def update_output_gradient(self):
        """计算近似器输出关于参数的梯度"""
        pass

    @abc.abstractmethod
    def train(self):
        """更新近似器

        需要在子类中实现
        """
        pass

    @abc.abstractmethod
    def predict(self):
        """基于近似器预测奖励

        需要在子类中实现
        """
        pass

    def update_confidence_bounds(self):
        """更新所有手臂的置信上界及相关量

        UCB核心步骤：
        1. 计算当前参数下每个手臂的梯度
        2. 基于梯度计算探索奖励（置信项）
        3. 预测每个手臂的奖励
        4. 合并为最终的置信上界
        """
        self.update_output_gradient()

        # UCB探索奖励 = 置信乘数 * sqrt(梯度^T * A^{-1} * 梯度)
        self.exploration_bonus[self.iteration] = np.array(
            [
                self.confidence_multiplier * np.sqrt(
                    np.dot(self.grad_approx[a], np.dot(self.A_inv[a], self.grad_approx[a].T))) for a in self.bandit.arms
            ]
        )

        # 预测每个手臂的奖励
        self.predict()

        # 置信上界 = 预测奖励 + 探索奖励
        self.upper_confidence_bounds[self.iteration] = self.mu_hat[self.iteration] + self.exploration_bonus[
            self.iteration]

    def update_A_inv(self):
        """使用Sherman-Morrison公式更新A_inv矩阵

        这是一个高效的秩1更新，避免了重新计算矩阵逆
        """
        self.A_inv[self.action] = inv_sherman_morrison(
            self.grad_approx[self.action],
            self.A_inv[self.action]
        )

    def run(self):
        """运行老虎机算法的一轮完整过程

        对每个时间步：
        1. 更新置信上界
        2. 选择动作
        3. 训练模型
        4. 更新探索指标
        5. 计算遗憾
        6. 记录日志
        """

        postfix = {
            'total regret': 0.0,
            '% optimal arm': 0.0,
        }

        total_regret = np.zeros(self.bandit.T + 1)
        timer = np.zeros(self.bandit.T)

        with tqdm(total=self.bandit.T, postfix=postfix) as pbar:
            for t in range(self.bandit.T):
                start_time = time.time()
                # 基于时刻t观察到的特征更新所有手臂的置信度
                self.update_confidence_bounds()
                # 选择具有最高置信上界的手臂
                self.action = self.sample_action()
                self.actions[t] = self.action
                # 更新近似器
                if t % self.train_every == 0:
                    self.train()
                # 更新探索指标 A_inv
                self.update_A_inv()
                # 计算遗憾（最优奖励 - 实际获得的奖励）
                self.regrets[t] = self.bandit.best_rewards_oracle[t] - self.bandit.rewards[t, self.action]
                total_regret[t] += self.regrets[t]
                # 迭代计数器加1
                self.iteration += 1
                end_time = time.time()

                # 记录日志
                csv_row = [str(t + 1), str(self.regrets[t]), str(total_regret[t]), str(end_time - start_time)]
                with open(self.log_output_dir, 'a+') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(csv_row)

                # 更新进度条显示的统计信息
                postfix['total regret'] += self.regrets[t]
                n_optimal_arm = np.sum(
                    self.actions[:self.iteration] == self.bandit.best_actions_oracle[:self.iteration]
                )
                postfix['% optimal arm'] = '{:.2%}'.format(n_optimal_arm / self.iteration)

                if t % self.throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(self.throttle)

        return total_regret, timer


class NeuralUCB(UCB):
    """Neural UCB (神经网络上置信界) 算法

    Neural UCB 是一种结合了深度学习与UCB框架的上下文老虎机算法。
    它使用神经网络来近似奖励函数，并通过UCB机制平衡探索与利用。

    核心思想：
    - 使用神经网络 f(x;θ) 近似奖励函数
    - 计算梯度 ∇_θ f(x;θ) 用于构建置信上界
    - UCB = 预测奖励 + 探索奖励（基于梯度信息矩阵）
    """

    def __init__(self,
                 bandit: ContextualBandit,         # 老虎机环境
                 hidden_size=20,                   # 隐藏层大小
                 n_layers=2,                       # 神经网络层数
                 reg_factor=1.0,                   # L2正则化因子
                 delta=0.01,                       # 置信参数
                 confidence_scaling_factor=-1.0,  # 置信缩放因子
                 training_window=100,               # 训练窗口大小
                 p=0.0,                            # Dropout概率
                 learning_rate=0.01,               # 学习率
                 epochs=1,                         # 训练轮数
                 train_every=1,                    # 训练频率
                 throttle=1,                       # 进度条更新频率
                 use_cuda=False,                   # 是否使用CUDA
                 log_output_dir='./'               # 日志目录
                 ):

        # 隐藏层神经元数量
        self.hidden_size = hidden_size
        # 神经网络层数
        self.n_layers = n_layers

        # 训练缓冲区大小（使用最近的数据进行训练）
        self.training_window = training_window

        # 神经网络参数
        self.learning_rate = learning_rate
        self.epochs = epochs

        # CUDA设置
        self.use_cuda = use_cuda
        if self.use_cuda:
            raise Exception(
                'Not yet CUDA compatible : TODO for later (not necessary to obtain good results')
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')

        # Dropout率
        self.p = p

        # 神经网络模型
        self.model = Model(input_size=bandit.n_features,
                           hidden_size=self.hidden_size,
                           n_layers=self.n_layers,
                           p=self.p
                           ).to(self.device)
        # Adam优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 调用父类UCB的初始化方法
        super().__init__(bandit,
                         reg_factor=reg_factor,
                         confidence_scaling_factor=confidence_scaling_factor,
                         delta=delta,
                         throttle=throttle,
                         train_every=train_every,
                         log_output_dir=log_output_dir
                         )

    @property
    def approximator_dim(self):
        """返回网络中所有可训练参数的总维度

        这等于神经网络中所有权重矩阵的元素总数
        """
        return sum(w.numel() for w in self.model.parameters() if w.requires_grad)

    @property
    def confidence_multiplier(self):
        """置信乘数

        返回置信上界计算中的缩放因子
        """
        return self.confidence_scaling_factor

    def update_output_gradient(self):
        """计算网络预测关于网络权重的梯度

        这是Neural UCB的核心：计算每个手臂特征输入时的输出梯度。
        这个梯度用于构建置信上界中的探索奖励项。
        """
        for a in self.bandit.arms:
            # 获取当前时刻手臂a的特征
            x = torch.FloatTensor(
                self.bandit.features[self.iteration, a].reshape(1, -1)
            ).to(self.device)

            # 清除之前的梯度
            self.model.zero_grad()
            # 前向传播
            y = self.model(x)
            # 反向传播，计算梯度
            y.backward()

            # 收集所有可训练参数的梯度，展平并拼接
            # 除以 sqrt(hidden_size) 用于归一化
            self.grad_approx[a] = torch.cat(
                [w.grad.detach().flatten() / np.sqrt(self.hidden_size) for w in self.model.parameters() if
                 w.requires_grad]
            ).to(self.device)

    def reset(self):
        """重置内部估计变量

        重新初始化所有UCB相关的状态变量
        """
        self.reset_upper_confidence_bounds()
        self.reset_regrets()
        self.reset_actions()
        self.reset_A_inv()
        self.reset_grad_approx()
        self.iteration = 0

    def train(self):
        """训练神经近似器

        使用最近training_window个时间步的数据训练神经网络。
        使用MSE损失函数进行监督学习。
        """
        # 获取最近training_window个时间步的数据索引
        iterations_so_far = range(np.max([0, self.iteration - self.training_window]), self.iteration + 1)
        actions_so_far = self.actions[np.max([0, self.iteration - self.training_window]):self.iteration + 1]

        # 准备训练数据
        x_train = torch.FloatTensor(self.bandit.features[iterations_so_far, actions_so_far]).to(self.device)
        y_train = torch.FloatTensor(self.bandit.rewards[iterations_so_far, actions_so_far]).squeeze().to(self.device)

        # 训练模式
        self.model.train()
        # 多个epoch训练
        for _ in range(self.epochs):
            y_pred = self.model.forward(x_train).squeeze()
            loss = nn.MSELoss()(y_train, y_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self):
        """预测奖励

        使用当前神经网络模型预测所有手臂的奖励
        """
        # 评估模式
        self.model.eval()
        # 对当前时刻所有手臂的特征进行预测
        self.mu_hat[self.iteration] = self.model.forward(
            torch.FloatTensor(self.bandit.features[self.iteration]).to(self.device)
        ).detach().squeeze()