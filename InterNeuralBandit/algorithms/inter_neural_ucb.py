import numpy as np
import math
import torch
from torch import nn
from torch import autograd

from base.agent import Agent
from base.reward import Reward

from HybridNeuralBandit.model import Model
from HybridNeuralBandit.model import CNN
from HybridNeuralBandit.data_handling import HistoryData
from HybridNeuralBandit.config import parse_args
from HybridNeuralBandit.algorithms.ucb import LinearPayoffUCB


class InterNeuralUCB(Agent):

    def __init__(self, historical_data: HistoryData, args: parse_args, use_cnn=False):
        self.historical_data = historical_data
        self.args = args
        self.model = Model(self.args.dim, self.args.latent_shape)
        if use_cnn:
            self.model = CNN()
        self.init_lin_ucb()

    def init_lin_ucb(self):
        self.lin_ucb = LinearPayoffUCB(np.array([[i for i in range(self.args.latent_shape)]]))

    def pre_training(self):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.offline_lr, weight_decay=0.0)
        for epoch in range(self.args.offline_max_epoch):
            assert self.args.offline_batch_size <= self.args.historical_data_size
            optimizer.zero_grad()
            sampled_data = self.historical_data.sample(self.args.offline_batch_size, self.args.feature_cols, self.args.reward_col)
            feature = sampled_data[:, self.args.feature_cols]
            reward = sampled_data[:, self.args.reward_col]
            # fix theta, train the intial value of f
            linpara = np.array([math.sqrt(self.args.latent_shape) for _ in range(self.args.latent_shape)])
            out = self.model(autograd.Variable(torch.FloatTensor(feature))).matmul(torch.tensor(linpara, dtype=torch.float))
            print(out.shape)
            #print(out)
            loss = criterion(out, autograd.Variable(torch.FloatTensor(reward)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(loss)

    def offline_training(self):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.offline_lr, weight_decay=0.0)
        for epoch in range(self.args.offline_max_epoch):
            assert self.args.offline_batch_size <= self.args.historical_data_size
            optimizer.zero_grad()
            sampled_data = self.historical_data.sample(self.args.offline_batch_size, self.args.feature_cols,
                                                       self.args.reward_col)
            feature = sampled_data[:, self.args.feature_cols]
            reward = sampled_data[:, self.args.reward_col]
            # fix theta, train f
            out = self.model(autograd.Variable(torch.FloatTensor(feature))).matmul(torch.tensor(self.lin_ucb.linpara, dtype=torch.float))
            #print(out)
            loss = criterion(out, autograd.Variable(torch.FloatTensor(reward)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #x = loss.detach().cpu().numpy()
            #print(x)

    def online_training(self, contexts, rewards):
        # fix f, train theta
        latent_features = self.model(autograd.Variable(torch.FloatTensor(contexts))).detach().cpu().numpy()
        #print(latent_features)
        self.lin_ucb.set_context(latent_features)
        action = self.lin_ucb.pick_action(observation=None)
        #print(self.lin_ucb.linpara)
        #print(action.actions)
        #print(rewards[action.actions])
        #print(self.lin_ucb.context[0, action.actions] * rewards[action.actions])
        self.lin_ucb.update_observation(action, Reward(rewards=rewards[action.actions]), observation=None)
        return action



