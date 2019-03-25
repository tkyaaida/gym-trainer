#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from typing import Tuple, List
from logging import getLogger
from gym_trainer.agents.dqn_agent import DQNAgent
from gym_trainer.helpers.replay_memory import Transition


logger = getLogger(__name__)


class DQNInteraction:
    def __init__(self, agent: DQNAgent, env):
        self.agent = agent
        self.env = env

    def run_episode_train(self, i_epoch: int) -> Tuple[float, List[Transition]]:
        reward_episode = 0
        data = []

        obs = self.env.reset()
        action = self.agent.step(obs, i_epoch)

        while True:
            next_obs, reward, done, _ = self.env.step(action)
            reward_episode += reward

            data.append(Transition(obs, action, next_obs, reward, done))

            if done:
                break

            obs = next_obs
            action = self.agent.step(obs, i_epoch)

        return reward_episode, data

    def run_episode_eval(self) -> float:
        reward_episode = 0

        obs = self.env.reset()
        action = self.agent.step_inference(obs)

        while True:
            obs, reward, done, _ = self.env.step(action)
            reward_episode += reward
            action = self.agent.step_inference(obs)

            if done:
                break

        return reward_episode
