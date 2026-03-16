# -*- coding: utf-8 -*-
"""
Create Synthetic data.
"""

import numpy.random as rd
import numpy as np
import time

class CreateData(object):
    def __init__(self, users, arms, dims, seed=int(time.time())):
        self.users = users
        self.arms = arms
        self.dims = dims
        self.data_rand = rd.RandomState(seed)
        self.contexts = np.ones((self.users, self.arms, self.dims), dtype=np.float)

    # ---- create a matrix of shape users*arms*dims (currently users should always set to 1) ----
    def data(self, mean, var):
        assert len(mean) == self.dims
        assert len(var) == self.dims

        res = []
        mean = np.array(mean)
        var = np.array(var)
        for i in range(self.users):
            res.append(self.data_rand.normal(mean, var, (self.arms, self.dims)))
        return np.array(res)


    def get_synthetic_context(self, args):
        """
        Generate Synthetic Contexts via given mean and variance
        """
        #create_data = CreateData(args.num_of_users, args.num_of_arms, args.dim)
        mean = [0.2, 0.9, 0.5, 3, 1.1, 0.9, 2, 2.5, 1.6, 1.8] * int(self.dims / 10)
        var = [3, 2, 4, 3, 3.5, 5.5, 5, 3.5, 5, 3.5] * int(self.dims / 10)
        context_gen = self.data(mean, var)
        # normalize
        ctx_norm = np.max(np.sqrt(np.sum(context_gen * context_gen, 2)), 1)
        for idx in range(self.users):
            context_gen[idx] = context_gen[idx] / ctx_norm[idx]
        self.contexts = context_gen

    def sample_spherical(self):
        """
        Generate Synthetic Contexts via randomly sampling from unit ball. Number of users = 1
        """
        vec = np.random.randn(self.dims, self.arms)
        vec /= np.linalg.norm(vec, axis=0)
        self.contexts = vec.T
