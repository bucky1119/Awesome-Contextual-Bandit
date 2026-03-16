""" assortment environments. """

import numpy as np

from base.environment import Environment
from base.action import Action
from base.reward import Reward
import torch
from torch import nn
from torch import autograd
from oracle_generator.data import FileData
from semiUCB.observation_context import *
import time


class ContextEnvironment(Environment):

    """Environments of contextual bandit."""

    def __init__(self, context, update_reward=True):
        """Initialize the environment."""
        assert isinstance(context, np.ndarray)
        assert len(np.shape(context)) == 2 or len(np.shape(context)) == 3
        if len(np.shape(context)) == 2:
            self.context = np.array([context.copy()])
        else:
            self.context = context.copy()
        self.cnum, self.n, self.d = np.shape(self.context)

        self.cid_rand = np.random.RandomState()
        self.reward_rand = np.random.RandomState()

        self.cid = 0 # there is only one customer
        self.reward = None
        self.optimal_reward = None
        self.optimal_action = None
        self.rewards = {}
        self.t = 0

        self.cid = self._update_cid()
        if update_reward:
            self._update_reward()
            self.optimal_action = np.argsort(self.reward)[-1]
            self.optimal_reward = self.reward[self.optimal_action]
            self.rewards[self.cid]=self.reward.copy()

    def _update_cid(self):
        return 0

    def _update_reward(self):
        """update the reward of corresponding context id"""
        pass

    def get_observation(self):
        """Returns an observation from the environment."""
        return ContextObservation(t=self.t, cid=self.cid)

    def get_optimal_reward(self):
        """Returns the optimal possible reward for the environment at that point."""
        self.optimal_action = int(np.argsort(self.reward)[-1])
        self.optimal_reward = self.reward[self.optimal_action]
        return self.optimal_reward

    def get_expected_reward(self, action):
        """Gets the expected reward of an action."""
        assert isinstance(action, Action)
        return self.reward[action.actions[0]]

    def get_stochastic_reward(self, action):
        """Gets a stochastic reward for the action."""
        pass

    def advance(self, action, reward):
        """Updating the environment (useful for nonstationary bandit)."""
        tmp_cid = self._update_cid()
        if tmp_cid != self.cid:
            self.cid = tmp_cid
            self.reward = self.rewards.get(self.cid, None)
            if self.reward is None:
                self._update_reward()
                # self.rewards[self.cid]=self.reward.copy()
            self.optimal_action = np.argsort(self.reward)[-1]
            self.optimal_reward = self.reward[self.optimal_action]
        self.t += 1


class LinearPayoffEnvironment(ContextEnvironment):
    """ time-invariant linear payoff environment"""

    def __init__(self, context, linpara, update_reward=True):
        super(LinearPayoffEnvironment, self).__init__(context, update_reward=False)
        assert isinstance(linpara, np.ndarray)
        assert len(linpara)==self.d
        self.linpara = linpara.copy()
        if update_reward:
            self._update_reward()
            self.optimal_action = np.argsort(self.reward)[-1]
            self.optimal_reward = self.reward[self.optimal_action]

    def _update_reward(self):
        """Calculate the reward for each arm
            - non-linear
        """
        self.reward = self.context[self.cid].dot(self.linpara)

    def get_stochastic_reward(self, action):
        """Gets a stochastic reward for the action.
            - expected reward + Gaussian noise
        """
        assert isinstance(action, Action)
        rval = self.reward[action.actions]+self.reward_rand.standard_normal()*0.001
        srwd = Reward(total_reward=np.sum(rval), rewards=rval)
        return srwd


class NonLinearCosPayoffEnvironment(ContextEnvironment):
    """ time-invariant nonlinear payoff environment
        reward models:
            1. r = 10(x^T a)^2 ,
            2. r = x^T A^T A x
        --> 3. r = cos(3x^T a)
                a is randomly generated from uniform distribution over unit ball.
            4. r = log(10 x^T a)
    """

    def __init__(self, context, gen_para, update_reward=True):
        super(NonLinearCosPayoffEnvironment, self).__init__(context, update_reward=False)
        assert isinstance(gen_para, np.ndarray)
        assert len(gen_para)==self.d
        self.gen_para = gen_para.copy()
        if update_reward:
            self._update_reward()
            self.optimal_action = np.argsort(self.reward)[-1]
            self.optimal_reward = self.reward[self.optimal_action]

    def _update_reward(self):
        """Calculate the reward for each arm
            - cosine reward
        """
        self.reward = np.cos(self.context[self.cid].dot(self.gen_para)*3)


    def get_stochastic_reward(self, action):
        """Gets a stochastic reward for the action.
            - expected reward + Gaussian noise
        """
        assert isinstance(action, Action)
        rval = self.reward[action.actions] + self.reward_rand.standard_normal()*0.001
        srwd = Reward(total_reward=np.sum(rval), rewards=rval)
        return srwd


class NonLinearSquarePayoffEnvironment(ContextEnvironment):
    """ time-invariant nonlinear payoff environment
        reward models:
        --> 1. r = 10(x^T a)^2
                a is randomly generated from uniform distribution over unit ball.
            2. r = x^T A^T A x
            3. r = cos(3x^T a)
            4. r = log(10 x^T a)
    """

    def __init__(self, context, gen_para, update_reward=True):
        super(NonLinearSquarePayoffEnvironment, self).__init__(context, update_reward=False)
        assert isinstance(gen_para, np.ndarray)
        assert len(gen_para)==self.d
        self.gen_para = gen_para.copy()
        if update_reward:
            self._update_reward()
            self.optimal_action = np.argsort(self.reward)[-1]
            self.optimal_reward = self.reward[self.optimal_action]

    def _update_reward(self):
        """Calculate the reward for each arm
            - square reward
        """
        self.reward = 10 * np.square(self.context[self.cid].dot(self.gen_para))


    def get_stochastic_reward(self, action):
        """Gets a stochastic reward for the action.
            - expected reward + Gaussian noise
        """
        assert isinstance(action, Action)
        rval = self.reward[action.actions] + self.reward_rand.standard_normal()*0.001
        srwd = Reward(total_reward=np.sum(rval), rewards=rval)
        return srwd


class NonLinearQuadPayoffEnvironment(ContextEnvironment):
    """ time-invariant nonlinear payoff environment
        reward models:
            1. r = 10(x^T a)^2
        --> 2. r = x^T A^T A x
                each entry of A (d x d) is randomly generated from N(0, 1)
            3. r = cos(3x^T a)
            4. r = log(10 x^T a)
    """

    def __init__(self, context, gen_para, update_reward=True):
        super(NonLinearQuadPayoffEnvironment, self).__init__(context, update_reward=False)
        assert isinstance(gen_para, np.ndarray)
        self.gen_para = gen_para.copy()
        if update_reward:
            self._update_reward()
            self.optimal_action = np.argsort(self.reward)[-1]
            self.optimal_reward = self.reward[self.optimal_action]

    def _update_reward(self):
        """Calculate the reward for each arm
            - quadratic reward
        """
        A = self.gen_para.T
        A /= np.linalg.norm(A, axis=0)
        A = A.T
        #prod = self.context[self.cid].dot(A.T).dot(A).dot(self.context[self.cid].T)
        prod = self.context[self.cid].dot(A).dot(self.context[self.cid].T)
        self.reward = np.diagonal(prod)

    def get_stochastic_reward(self, action):
        """Gets a stochastic reward for the action.
            - expected reward + Gaussian noise
        """
        assert isinstance(action, Action)
        rval = self.reward[action.actions]+self.reward_rand.standard_normal()*0.001
        srwd = Reward(total_reward=np.sum(rval), rewards=rval)
        return srwd


class NonLinearLogPayoffEnvironment(ContextEnvironment):
    """ time-invariant nonlinear payoff environment
        reward models:
            1. r = 10(x^T a)^2
            2. r = x^T A^T A x
                each entry of A (d x d) is randomly generated from N(0, 1)
            3. r = cos(3x^T a)
        --> 4. r = log(10 x^T a)
    """
    def __init__(self, context, gen_para, update_reward=True):
        super(NonLinearLogPayoffEnvironment, self).__init__(context, update_reward=False)
        assert isinstance(gen_para, np.ndarray)
        self.gen_para = gen_para.copy()
        if update_reward:
            self._update_reward()
            self.optimal_action = np.argsort(self.reward)[-1]
            self.optimal_reward = self.reward[self.optimal_action]

    def _update_reward(self):
        """Calculate the reward for each arm
            - quadratic reward
        """
        self.reward = np.log10(np.abs(10 * self.context[self.cid].dot(self.gen_para)))

    def get_stochastic_reward(self, action):
        """Gets a stochastic reward for the action.
            - expected reward + Gaussian noise
        """
        assert isinstance(action, Action)
        rval = self.reward[action.actions]+self.reward_rand.standard_normal()*0.001
        srwd = Reward(total_reward=np.sum(rval), rewards=rval)
        return srwd


class NonLinearExpPayoffEnvironment(ContextEnvironment):
    """
    time-invariant nonlinear payoff environment
    r(x) = e^(x^T a)
    """
    def __init__(self, context, gen_para, update_reward=True):
        super(NonLinearExpPayoffEnvironment, self).__init__(context, update_reward=False)
        assert isinstance(gen_para, np.ndarray)
        self.gen_para = gen_para.copy()
        if update_reward:
            self._update_reward()
            self.optimal_action = np.argsort(self.reward)[-1]
            self.optimal_reward = self.reward[self.optimal_action]

    def _update_reward(self):
        """Calculate the reward for each arm
            - quadratic reward
        """
        self.reward = np.exp(self.context[self.cid].dot(self.gen_para))

    def get_stochastic_reward(self, action):
        """Gets a stochastic reward for the action.
            - expected reward + Gaussian noise
        """
        assert isinstance(action, Action)
        rval = self.reward[action.actions]+self.reward_rand.standard_normal()*0.001
        srwd = Reward(total_reward=np.sum(rval), rewards=rval)
        return srwd


class RealDataEnvironment(ContextEnvironment):
    """
    Environment for real-world dataset
    """
