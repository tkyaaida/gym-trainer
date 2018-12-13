#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import chainer
from chainer.backends import cuda
from chainer import Chain
import chainer.links as L
import chainer.functions as F
from chainer.distributions import Categorical


class CategoricalPolicy(Chain):
    """Choose discrete actions"""
    def __init__(self, dim_input, dim_output, dim_hidden=100, dropout_ratio=0.5):
        super(CategoricalPolicy, self).__init__()

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dropout_ratio = dropout_ratio

        with self.init_scope():
            self.l1 = L.Linear(dim_input, dim_hidden, nobias=True)
            self.l2 = L.Linear(dim_hidden, dim_output, nobias=True)

    def forward(self, x):
        """

        Args:
            x: (batchsize, dim_input)

        Returns:

        """
        assert x.shape[1] == self.dim_input
        h = F.relu(F.dropout(self.l1(x), ratio=self.dropout_ratio))
        y = self.l2(h)
        return y


class ReinforceAgent:
    def __init__(self, dim_input, dim_output, device):
        self.policy = CategoricalPolicy(dim_input, dim_output)
        self.xp = self.policy.xp
        self.device = device

    def reset_inference(self, obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            obs = self.xp.array(obs, dtype=self.xp.float32)[None, :]
            if self.device >= 0:
                obs = cuda.to_gpu(obs, device=self.device)
            prob = F.softmax(self.policy(obs), axis=1)
            c = Categorical(prob)
            action = c.sample_n(1)
            return action.array[0, 0]

    def step_inference(self, obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            obs = self.xp.array(obs, dtype=self.xp.float32)[None, :]
            if self.device >= 0:
                obs = cuda.to_gpu(obs, device=self.device)
            prob = F.softmax(self.policy(obs), axis=1)
            c = Categorical(prob)
            action = c.sample_n(1)
            return action.array[0, 0]

    def calc_reward_to_go(self, rewards):
        """

        Args:
            rewards: (N, T)

        Returns:

        """
        N, T = rewards.shape
        xp = self.policy.xp
        reward_to_go = xp.flip(xp.cumsum(xp.flip(rewards, axis=1), axis=1), axis=1)
        baseline = xp.mean(reward_to_go, axis=0)
        advantage = reward_to_go - baseline
        return advantage.reshape((N*T, ))
