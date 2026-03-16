"""
Implementation for all UCB- and Thompson Sampling-like baselines:
=================================================================
1. LinUCB,
2. Thompson Sampling with linear reward,
3. Random projection,
4. KernelUCB.
"""

import time
import numpy as np

from sklearn import random_projection

from base.agent import Agent
from base.action import Action
from base.reward import Reward
from base.observation import Observation
from context.observation_context import ContextObservation

'''
NOTE: We do NOT need 'cid' as our paper does not consider personalized recommendation.
      We still preserve the parameter 'cid' for further extension
'''


class LinearPayoffBandit(Agent):
    def __init__(self, context):
        """Initialize the agent."""
        self.set_context(context)

        self.A = np.eye(self.d, self.d)
        self.b = np.zeros(self.d)
        self.invA = np.linalg.inv(self.A)
        self.linpara = self.invA.dot(self.b)
        self.reward = np.zeros(self.n)

        self.select_ls = []
        self.model_ratio = 0

    def set_context(self, context):
        assert isinstance(context, np.ndarray)
        assert len(np.shape(context)) == 2 or len(np.shape(context)) == 3
        # Convert context to the shape of users*arms*dims
        if len(np.shape(context)) == 2:
            self.context =  np.array([context.copy()])
        else:
            self.context =  context.copy()
        self.cnum, self.n, self.d = np.shape(self.context)

    def update_observation(self, action, reward, observation):
        """Add an observation to the records."""
        assert observation is None or isinstance(observation, Observation)

        assert isinstance(action, Action)
        assert isinstance(reward, Reward)
        context = self.context[0]
        xmat = context[action.actions].reshape(1,-1)
        self.A += xmat.T.dot(xmat)
        #self.b += context[action.actions].T.dot(reward.rewards)
        self.b += (context[action.actions] * reward.rewards)[0]
        #print(self.b)
        self.invA = np.linalg.inv(self.A)
        #print(self.invA)
        self.linpara = self.invA.dot(self.b)

    def pick_action(self, observation):
        """Select an action based upon the policy + observation."""
        assert observation is None or isinstance(observation, Observation)
        self.reward = self.context[0].dot(self.linpara)
        optimal_action = Action(actions=np.argsort(self.reward)[-1])
        return optimal_action


class LinearPayoffTS(LinearPayoffBandit):
    def __init__(self, context, var=0.3):
        """Initialize the agent."""
        super(LinearPayoffTS, self).__init__(context)
        self.smppara = np.zeros(self.d)
        self.var = var #*var
        self.theta_rand = np.random.RandomState()

    def pick_action(self, observation):
        """Select an action based upon the policy + observation."""
        assert observation is None or isinstance(observation, Observation)

        context = self.context[0]
        self.smppara = np.random.multivariate_normal(self.linpara, self.var*self.invA)
        self.reward = context.dot(self.smppara)
        optimal_action = Action(actions=np.argsort(self.reward)[-1])
        return optimal_action


class LinearPayoffUCB(LinearPayoffBandit):
    """
    Implementation of LinUCB.
    """
    def __init__(self, context, alpha=0.1, offline_scores=[]):
        """Initialize the agent."""
        super(LinearPayoffUCB, self).__init__(context)
        # self.smppara = np.zeros(self.d)
        self.alpha = alpha

    def pick_action(self, observation):
        """Select an action based upon the policy + observation."""
        assert observation is None or isinstance(observation, Observation)

        context = self.context[0]
        self.reward = context.dot(self.linpara) + self.alpha * np.sqrt(np.diag(context.dot(self.invA).dot(context.T)))
        #print(self.reward)
        x = np.argsort(self.reward)[-1]
        optimal_action = Action(actions=int(x))
        #print(optimal_action.actions)
        return optimal_action


class LinearPayoffUCBWithRP(Agent):
    def __init__(self, context, var=0.3, rp_dim = 10):
        """Initialize the agent."""
        self.set_context(context)

        self.rp_dim = rp_dim

        self.A = np.eye(self.rp_dim, self.rp_dim)
        self.b = np.zeros(self.rp_dim)
        self.invA = np.linalg.inv(self.A)
        self.linpara = self.invA.dot(self.b)
        self.reward = np.zeros(self.n)

        self.smppara = np.zeros(self.rp_dim)
        self.var = var  # *var
        self.theta_rand = np.random.RandomState()

        transformer = random_projection.GaussianRandomProjection(n_components=rp_dim, random_state=int(time.time()))
        X = np.random.rand(1, self.d)
        self.RP = transformer.fit(X).components_.T

    def set_context(self, context):
        assert isinstance(context, np.ndarray)
        assert len(np.shape(context)) == 2 or len(np.shape(context)) == 3
        # Convert context to the shape of users*arms*dims
        if len(np.shape(context)) == 2:
            self.context = np.array([context.copy()])
        else:
            self.context = context.copy()
        self.cnum, self.n, self.d = np.shape(self.context)


    def update_observation(self, action, reward, observation):
        """Add an observation to the records."""
        assert observation is None or isinstance(observation, Observation)
        cid = 0
        if isinstance(observation, ContextObservation):
            cid = observation.cid
        assert isinstance(action, Action)
        assert isinstance(reward, Reward)

        context = self.context[cid].dot(self.RP)
        # context = context / np.max(np.sqrt(np.sum(context * context, 1)))

        xmat = context[action.actions].reshape(1,-1)
        self.A += xmat.T.dot(xmat)
        # self.b += context[action.actions].T.dot(reward.rewards)
        self.b += (context[action.actions] * reward.rewards)[0]
        self.invA = np.linalg.inv(self.A)
        self.linpara = self.invA.dot(self.b)

    def pick_action(self, observation):
        """Select an action based upon the policy + observation."""
        assert observation is None or isinstance(observation, Observation)
        cid = 0
        if isinstance(observation, ContextObservation):
            cid = observation.cid

        context = self.context[cid].dot(self.RP)
        # context = context / np.max(np.sqrt(np.sum(context * context, 1)))

        self.reward = context.dot(self.linpara) + self.var * np.sqrt(np.diag(context.dot(self.invA).dot(context.T)))
        optimal_action = Action(actions=np.argsort(self.reward)[-1])
        return optimal_action


class KernelUCB(Agent):
    """
    Implementation of Kernel UCB in paper "Finite-Time Analysis of Kernelised Contextual Bandits".
    """
    def __init__(self, context, num_arms, gamma, eta, kernel):
        # hyperparameters
        self.set_context(context)
        self.gamma = gamma
        self.eta = eta
        self.kernel = kernel
        self.n = num_arms
        self.u = np.zeros(self.n)
        self.u[0] = 1
        self.y = []
        self.K = None
        self.Kinv = None
        self.opt_arms = None

    def set_context(self, context):
        assert isinstance(context, np.ndarray)
        assert len(np.shape(context)) == 2 or len(np.shape(context)) == 3
        # Convert context to the shape of users*arms*dims
        if len(np.shape(context)) == 2:
            self.context = np.array([context.copy()])
        else:
            self.context = context.copy()

    def pick_action(self, observation):
        assert observation is None or isinstance(observation, Observation)
        x = np.argsort(self.u)[-1]
        optimal_action = Action(actions=int(x))
        return optimal_action

    def update_kernel_matrix(self, action, reward):
        assert isinstance(action, Action)
        assert isinstance(reward, Reward)
        context = self.context[0, action.actions]
        #print(context)

        self.y.append(reward.rewards)

        if self.opt_arms is None:
            self.opt_arms = context
        else:
            self.opt_arms = np.concatenate((self.opt_arms, context), axis=0)

        if self.Kinv is None:
            self.Kinv = 1 / self.kernel(context, context) + self.gamma
            self.K = np.linalg.inv(self.Kinv)
        else:
            '''
            time = np.shape(self.opt_arms)[0]
            pre_opt_arms = self.opt_arms[0:time-1]
            b = self.kernel(np.array([pre_opt_arms[0]]), pre_opt_arms)
            t = 1
            while t < time - 1:
                b = np.concatenate((b, self.kernel(np.array([pre_opt_arms[t]]), pre_opt_arms)), axis=0)
                t += 1

            K22 = np.linalg.inv(self.kernel(context, context) + self.gamma - b.transpose().dot(self.Kinv).dot(b) + 1e-6)
            K11 = self.Kinv + K22.dot(self.Kinv).dot(b).dot(b.transpose()).dot(self.Kinv)
            K12 = -K22.dot(self.K).dot(b)
            K21 = -K22.dot(b.transpose()).dot(self.Kinv)
            self.Kinv = np.concatenate((np.concatenate((K11, K12), axis=1), np.concatenate((K21, K22), axis=1)), axis=0)
            self.K = np.linalg.inv(self.Kinv)
            '''
            time = np.shape(self.opt_arms)[0]
            K = self.kernel(np.array([self.opt_arms[0]]), self.opt_arms)
            t = 1
            while t < time:
                K = np.concatenate((K, self.kernel(np.array([self.opt_arms[t]]), self.opt_arms)), axis=0)
                t += 1
            self.K = K + np.identity(time) * self.gamma
            self.Kinv = np.linalg.inv(self.K)



    def evaluate_arms(self):
        context = self.context[0]
        sigma = np.empty(self.n)
        #print(np.array([context[0]]))
        #print(self.opt_arms)
        for idx in range(self.n):
            k = self.kernel(np.array([context[idx]]), self.opt_arms)
            #print(self.opt_arms)
            #print(k)
            #print(self.Kinv)
            #print(self.kernel(np.array([context[idx]]), np.array([context[idx]])) - k.dot(self.Kinv).dot(k.transpose()))
            sigma[idx] = np.sqrt(
                self.kernel(np.array([context[idx]]), np.array([context[idx]])) - k.dot(self.Kinv).dot(k.transpose()))
            self.u[idx] = k.dot(self.Kinv).dot(np.array([self.y]).transpose()) + self.eta * sigma[idx] / np.sqrt(
                self.gamma)
