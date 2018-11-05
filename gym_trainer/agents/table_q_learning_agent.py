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
        self.n_disc = n_disc  # 何段階に離散化するか

        self.initial_eps = 0.8
        self.step_size = 0.8
        self.gamma = 0.99

        self.q_values = np.zeros((n_disc**dim_obs, dim_action))
        self.state = None
        self.i_episode = 0

    def reset(self, observation):
        self.i_episode += 1
        self.state = self.get_state(observation)
        action = self.get_action()
        return action

    def step(self, observation, reward, action):
        state_prime = self.get_state(observation)

        # update params
        self.q_values[self.state, action] += \
            self.step_size * (reward + self.gamma * self._q_max(state_prime) - self.q_values[self.state, action])
        self.state = state_prime

        # choose action
        next_action = self.get_action()
        return next_action

    def step_inference(self, observation):
        state = self.get_state(observation)
        action = np.argmax(self.q_values[state, :])
        return action

    def get_action(self):
        """get action according to behavior policy (epsilon greedy)"""
        eps = get_decayed_param(self.initial_eps, 0.99, self.i_episode)
        action = epsilon_greedy(eps, list(self.q_values[self.state, :]))
        return action

    def get_state(self, observation):
        """return discretized state given observation and action

        Args:
            observation: raw value returned by environment
        """
        obs_disc = [
            int(
                np.digitize(
                    observation[i],
                    bins=np.linspace(self.obs_min[i], self.obs_max[i], self.n_disc-1)
                )
            ) for i in range(len(observation))]

        state = sum([x + i * (self.n_disc-1) for i, x in enumerate(obs_disc)])

        return state

    def _q_max(self, state):
        """return maximum q value given state"""
        q_max = np.max(self.q_values[state, :])
        return q_max
