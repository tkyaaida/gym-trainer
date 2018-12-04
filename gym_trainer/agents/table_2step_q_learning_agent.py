#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Q-learning agent for discrete state

import numpy as np
from gym_trainer.helpers.utils import epsilon_greedy, get_decayed_param


class Table2StepQLearningAgent:
    def __init__(self, dim_obs, dim_action, obs_min, obs_max, n_disc,
                 lr=0.1, gamma=0.99, eps_ini=0.8, eps_inf=0.05, eps_decay_rate=0.99):
        """

        Args:
            dim_obs (int): dimension of observation
            dim_action (int): dimension of action
            obs_min (list): list of min value for each dimension
            obs_max (list): list of max value for each dimension
            n_disc (int): number of discretization for each dimension
            lr (float): a learning rate
            gamma (float): a discount factor
            eps_ini (float): initial epsilon value
            eps_inf (float): stationary epsilon value
            eps_decay_rate (float): decay rate of epsilon value
        """
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        assert len(obs_min) == dim_obs
        assert len(obs_max) == dim_obs
        assert [x < 0 for x in obs_min] == [True] * len(obs_min)  # every element of obs_min is negative
        assert [0 < x for x in obs_max] == [True] * len(obs_max)  # every element of obs_max is positive
        self.obs_min = obs_min
        self.obs_max = obs_max
        assert n_disc >= 2
        self.n_disc = n_disc

        self.lr = lr
        self.gamma = gamma
        assert eps_ini + eps_inf <= 1.0
        self.eps_ini = eps_ini
        self.eps_inf = eps_inf
        self.eps_decay_rate = eps_decay_rate

        self.q_values = np.zeros(((2 * n_disc) ** dim_obs, dim_action))

    def reset(self, obs, i_episode):
        state = self.get_state(obs)
        action = self.get_action(state, i_episode)
        return action

    def reset_inference(self, obs):
        state = self.get_state(obs)
        action = self.get_action_inference(state)
        return action

    def step(self, obs, action, obs_next_next, reward, reward_next, i_episode):
        state = self.get_state(obs)
        state_next_next = self.get_state(obs_next_next)

        # update params
        self.q_values[state, action] += self.lr * (reward + reward_next +
                                                   self.gamma ** 2 * self._q_max(state_next_next) -
                                                   self.q_values[state, action])
        # choose action
        action_next_next = self.get_action(state_next_next, i_episode)
        return action_next_next

    def step_inference(self, obs):
        state = self.get_state(obs)
        action = np.argmax(self.q_values[state, :])
        return action

    def get_action(self, state, i_episode):
        """get action according to behavior policy (epsilon greedy)"""
        eps = get_decayed_param(self.eps_ini, self.eps_inf, self.eps_decay_rate, i_episode)
        action = epsilon_greedy(eps, list(self.q_values[state, :]))
        return action

    def get_action_inference(self, state):
        action = np.argmax(self.q_values[state, :])
        return action

    def get_state(self, obs):
        """return discretized state given observation and action"""
        # get list of discretized value for each dimension
        obs_disc = [
            int(np.digitize(obs[i], bins=self._get_bins(self.obs_min[i], self.obs_max[i])))
            for i in range(len(obs))
        ]
        # convert obs_disc to single integer that represents its state
        # e.g.) if self.n_disc = 3, each dimension is represented as integer from 0 to 5
        # thus, if discretized state is for example [0, 5, 3, 1], it will be converted to a single integer by
        # 0*6^0 + 5*6^1 + 3*6^2 + 1*6^3
        state = sum([x * (2 * self.n_disc) ** i for i, x in enumerate(obs_disc)])
        return state

    def _get_bins(self, obs_min, obs_max):
        """get bins to discretize

        length of bins == 2 * self.n_disc - 1
        total number of discrete state per dimension: 2 * self.n_disc

        Args:
            obs_min (float):
            obs_max (float):

        Returns:

        """
        if obs_min < 0 and obs_max > 0:
            b1 = np.linspace(obs_min, 0, self.n_disc)
            b2 = np.linspace(0, obs_max, self.n_disc)
            return np.hstack([b1, b2[1:]])
        else:
            bins = np.linspace(obs_min, obs_max, 2 * self.n_disc - 1)
            return bins

    def _q_max(self, state):
        """return maximum q value given state"""
        q_max = np.max(self.q_values[state, :])
        return q_max
