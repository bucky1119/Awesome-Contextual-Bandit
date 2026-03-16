import numpy as np

from base.agent import Agent
import random
import math
from base.action import Action
from base.reward import Reward
from base.observation import Observation
from context.observation_context import ContextObservation

class Exp3Bandit(Agent):
    def __init__(self, gamma, arm_nums):
        """Initialize the agent."""
        self.gamma = gamma
        self.arm_nums = arm_nums
        self.exposure = np.zeros(arm_nums)
        self.success = np.zeros(arm_nums)
        self.reward = np.zeros(arm_nums)
        self.weight = np.ones(arm_nums)
        self.select_ls = []
        self.model_ratio = 0

    def update_observation(self, action, reward, observation=None,  reward_min=0, reward_max=1):
        """Add an observation to the records."""
        assert observation is None or isinstance(observation, Observation)
        assert isinstance(action, Action)
        assert isinstance(reward, Reward)
        self.exposure[action.actions] += 1
        self.success[action.actions] += reward.rewards[0]
        arm_id = action.actions[0]
        r = reward.rewards[0]
        scaled_r = (r - reward_min) * 1.0 / (reward_max - reward_min)
        estimate_r = 1.0 * scaled_r / self.p[arm_id]
        gamma_t = self.gamma_function()
        self.weight[arm_id] *= math.exp(estimate_r * gamma_t * 1.0 / (self.arm_nums * 1.0))

    def gamma_function(self, t=None):
        return self.gamma
        #return min(1. / self.arm_nums, np.sqrt(np.log(self.arm_nums) / (t * self.arm_nums)))

    def draw(self, weights):
        # method1: bigger weight and easier to choose
        action = random.uniform(0, sum(weights))
        actionIndex = 0
        for weight in weights:
            action -= weight
            if action <= 0:
                return actionIndex
            actionIndex += 1

    def pick_action(self, observation):
        """Select an action based upon the policy + observation."""
        assert observation is None or isinstance(observation, Observation)
        weight_sum = float(np.sum(self.weight))
        gamma_t = self.gamma_function()
        self.p = tuple((1.0 - gamma_t) * (w / weight_sum) + (gamma_t / self.arm_nums) for w in self.weight)
        # print("exp3 distribution:", self.p)
        optimal_action = Action(actions=self.draw(self.p))
        return optimal_action

    def get_estimated_reward_probability(self):
        return self.p

    def get_success_reward_probability(self):
        #a = []
        #for i in range(len(self.exposure)):
        #    if self.exposure[i] != 0:
        #        a.append(self.success[i] / self.exposure[i])
        #    else:
        #        a.append(0)
        #return np.array(a)
        return self.exposure

    def get_regret_bound(self, cum_reward, t):
        gamma = min(1. / self.arm_nums, np.sqrt(np.log(self.arm_nums) / (t * self.arm_nums)))
        return (math.e-1)*gamma*cum_reward+(self.arm_nums*math.log(self.arm_nums))/ gamma

