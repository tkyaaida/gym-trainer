#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SARSA(λ) agent


import numpy as np
from gym_trainer.helpers.utils import epsilon_greedy, get_decayed_param


class TableSarsaLambdaAgent:
    def __init__(self, dim_obs, dim_action, obs_min, obs_max, n_disc,
                 lr=0.1, gamma=0.99, eps_ini=0.8, eps_inf=0.1, eps_decay_rate=0.99, _lambda=0.7):
        """

        Args:
            dim_obs (int): dimension of observation
            dim_action (int): dimension of action
            obs_min (list): list of min value for each dimension
            obs_max (list): list of max value for each dimension
            n_disc (int): number of discretization
            lr (float): learning rate
            gamma (float): discount factor
            eps_ini (float): initial epsilon value
            eps_inf (float): stationary epsilon value
            eps_decay_rate (float): decay rate for epsilon value
            _lambda (float): parameter for calculating λ-return
        """

        self.dim_obs = dim_obs
        self.dim_action = dim_action
        assert len(obs_min) == dim_obs
        assert len(obs_max) == dim_obs
        self.obs_min = obs_min
        self.obs_max = obs_max
        self.n_disc = n_disc

        self.lr = lr
        self.gamma = gamma
        assert eps_ini + eps_inf <= 1.0
        self.eps_ini = eps_ini
        self.eps_inf = eps_inf
        self.eps_decay_rate = eps_decay_rate
        self._lambda = _lambda

        self.q = np.zeros(((2 * n_disc) ** dim_obs, dim_action))
        self.e = np.zeros(((2 * n_disc) ** dim_obs, dim_action))  # eligibility trace

    def reset(self, observation, i_episode):
        self.e = np.zeros(((2*self.n_disc)**self.dim_obs, self.dim_action))
        state = self.get_state(observation)
        action = self.get_action(state, i_episode)
        return action

    def step(self, obs, action, obs_next, reward, i_episode):
        """agent step when training

        Args:
            obs: previous observation
            action: action agent took given obs
            obs_next: next observation agent got
            reward: reward agent got
            i_episode (int): index of episode

        Returns:

        """
        state = self.get_state(obs)
        state_next = self.get_state(obs_next)
        action_next = self.get_action(state_next, i_episode)
        delta = reward + self.gamma * self.q[state_next, action_next] - self.q[state, action]
        self.e[state, action] += 1

        for i in range(self.n_disc**self.dim_obs):
            for j in range(self.dim_action):
                self.q[i, j] += self.lr * delta * self.e[i, j]
                self.e[i, j] = self.gamma * self._lambda * self.e[i, j]
        return action_next

    def step_inference(self, obs):
        state = self.get_state(obs)
        action = self.get_action_inference(state)
        return action

    def get_action(self, state, i_episode):
        # follow ε-greedy policy
        eps = get_decayed_param(self.eps_ini, self.eps_inf, self.eps_decay_rate, i_episode)
        action = epsilon_greedy(eps, list(self.q[state, :]))
        return action

    def get_action_inference(self, state):
        action = np.argmax(self.q[state, :])
        return action


