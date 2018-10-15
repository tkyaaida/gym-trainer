#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Q-learning Agent
#
# implementation of Q-learning with linear function approximation
#
# action: discrete
# state: continuous

import numpy as np
from gym_trainer.helpers.utils import epsilon_greedy, get_decayed_param


class LinearQLearningAgent:
    def __init__(self, dim_obs, dim_action):
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.dim_state = 1 + dim_obs * dim_action

        self.w = (np.random.rand(self.dim_state) - 0.5) / 50

        self.initial_eps = 0.8
        self.step_size = 0.01
        self.gamma = 0.99

    def reset(self, observation):
        """given initial observation of env, return initial action to take."""
        action = self.get_action(observation, 0)
        return action

    def step(self, observation, reward, i_episode, action):
        """A single step to perform"""
        # update parameter
        q = reward + self.gamma * self._q_max(observation)
        self.w = self.w - self.step_size * (q - self._q_hat(observation, action)) * self.get_state(observation, action)

        # decide next action
        next_action = self.get_action(observation, i_episode)
        return next_action

    def step_inference(self, observation):
        q_values = self._calc_q_values(observation)
        # select action according to greedy policy
        return np.argmax(np.array(q_values))

    def get_state(self, observation, action):
        """update state given observation"""
        bias = 1
        state = np.zeros(self.dim_state)
        state[0] = bias
        state[self.dim_obs*action+1:self.dim_obs*(action+1)+1] = observation
        return state

    def get_action(self, observation, i_episode):
        """get action according to behavior policy (which is epsilon greedy policy)"""
        q_values = self._calc_q_values(observation)
        eps = get_decayed_param(self.initial_eps, 0.99, i_episode)
        action = epsilon_greedy(eps, q_values)
        return action

    def _q_hat(self, observation, action):
        """return estimate of Q value given observation and action"""
        state = self.get_state(observation, action)
        return np.dot(self.w, state)

    def _q_max(self, observation):
        """return max Q(s, a)"""
        q_values = self._calc_q_values(observation)
        return max(q_values)

    def _calc_q_values(self, observation):
        """calc q values"""
        q_values = [self._q_hat(observation, action) for action in range(self.dim_action)]
        return q_values
