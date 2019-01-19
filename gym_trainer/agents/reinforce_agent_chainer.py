#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import numpy as np
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
    def __init__(self, dim_input, dim_output, optimizer, device):
        self.policy = CategoricalPolicy(dim_input, dim_output)
        # self.xp = self.policy.xp
        self.optimizer = optimizer
        self.device = device

        self.optimizer.setup(self.policy)

    def reset_inference(self, obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            obs = self.policy.xp.array(obs, dtype=self.policy.xp.float32)[None, :]
            if self.device >= 0:
                obs = cuda.to_gpu(obs, device=self.device)
            prob = F.softmax(self.policy(obs), axis=1)
            c = Categorical(prob)
            action = c.sample_n(1).array
            assert action.shape == (1, 1)
            if self.device >= 0:
                action = cuda.to_cpu(action)
            return action[0, 0]

    def step_inference(self, obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            obs = self.policy.xp.array(obs, dtype=self.policy.xp.float32)[None, :]
            if self.device >= 0:
                obs = cuda.to_gpu(obs, device=self.device)
            prob = F.softmax(self.policy(obs), axis=1)
            c = Categorical(prob)
            action = c.sample_n(1).array
            assert action.shape == (1, 1)
            if self.device >= 0:
                action = cuda.to_cpu(action)
            return action[0, 0]

    def step(self, trajectories):
        n_rollout = len(trajectories)
        n_step = len(trajectories[0])

        obs, actions, rewards = self.convert(trajectories)
        ce = F.softmax_cross_entropy(self.policy.forward(obs), actions, reduce='no')
        assert ce.shape == (n_rollout * n_step, )
        reward_to_go = self.calc_reward_to_go(rewards)
        loss = ce * reward_to_go
        loss = loss.reshape((n_rollout, n_step))
        loss = F.mean(F.sum(loss, axis=1))  # summation along step axis
        self.policy.cleargrads()
        loss.backward()
        self.optimizer.update()

        return loss.array

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

    def convert(self, trajectories):
        """convert trajectories (list of trajectories) into ndarray and send to device"""
        obs = []
        actions = []
        rewards = []

        for trajectory in trajectories:
            for transition in trajectory:
                obs.append(transition['obs'])
                actions.append(transition['action'])
                rewards.append(transition['reward'])

        n_rollout = len(trajectories)
        n_step = len(trajectories[0])

        obs = np.vstack(obs)
        obs = obs.astype(np.float32)
        assert obs.shape[0] == n_rollout * n_step
        actions = np.array(actions, dtype=np.int32)
        assert actions.shape == (n_rollout * n_step, )
        rewards = np.array(rewards, dtype=np.float32).reshape((n_rollout, n_step))

        if self.device >= 0:
            obs = cuda.to_gpu(obs, device=self.device)
            actions = cuda.to_gpu(actions, device=self.device)
            rewards = cuda.to_gpu(rewards, device=self.device)

        return obs, actions, rewards
