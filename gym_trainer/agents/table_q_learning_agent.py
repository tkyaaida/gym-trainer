#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Q-learning agent for discrete state

import numpy as np
from gym_trainer.helpers.utils import epsilon_greedy, get_decayed_param


class TableQLearningAgent:
    def __init__(self, dim_obs, dim_action, obs_min, obs_max, n_disc):
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.obs_min = obs_min
        self.obs_max = obs_max
        self.n_disc = n_disc

        self.initial_eps = 0.8
        self.step_size = 0.8
        self.gamma = 0.99

        self.q_values = np.zeros(((n_disc+1)**dim_obs, dim_action))
        self.state = None

    def reset(self, observation):
        self.state = self.get_state(observation)
        action = self.get_action(0)
        return action

    def step(self, observation, reward, i_episode, action):
        state_prime = self.get_state(observation)

        # update params
        self.q_values[self.state, action] += \
            self.step_size * (reward + self.gamma * self._q_max(state_prime) - self.q_values[self.state, action])
        self.state = state_prime

        # choose action
        next_action = self.get_action(i_episode)
        return next_action

    def step_inference(self, observation):
        state = self.get_state(observation)
        return np.argmax(self.q_values[state, :])

    def get_action(self, i_episode):
        """get action according to behavior policy (epsilon greedy)"""
        eps = get_decayed_param(self.initial_eps, 0.99, i_episode)
        action = epsilon_greedy(eps, self.q_values)
        if action % 2 == 0:
            action = 0
        else:
            action = 1
        return action

    def get_state(self, observation):
        """return discretized state given observation and action"""
        obs_disc = [
            int(
                np.digitize(
                    observation[i],
                    bins=np.linspace(self.obs_min[i], self.obs_max[i], self.n_disc)
                )
            ) for i in range(len(observation))]

        state = sum([x + i * self.n_disc for i, x in enumerate(obs_disc)])

        return state

    def _q_max(self, state):
        """return maximum q value given state"""
        q_max = np.max(self.q_values[state, :])
        return q_max
