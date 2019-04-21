#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from typing import Tuple, List
from logging import getLogger
from gym_trainer.agents.ddpg_agent import DDPGAgent
from gym_trainer.helpers.replay_memory import Transition


logger = getLogger(__name__)


class DDPGInteraction:
    def __init__(self, agent: DDPGAgent, env):
        self.agent = agent  # type: DDPGAgent
        self.env = env

    def run_episode_train(self) -> Tuple[float, List[Transition]]:
        reward_episode = 0
        data = []

        obs = self.env.reset()
        self.agent.reset()
        action = self.agent.step(obs)

        while True:
            next_obs, reward, done, _ = self.env.step(action)
            reward_episode += reward

            data.append(Transition(obs, action, next_obs, reward, done))

            if done:
                break

            obs = next_obs
            action = self.agent.step(obs)

        return reward_episode, data

    def run_episode_eval(self) -> float:
        reward_episode = 0

        obs = self.env.reset()
        self.agent.reset()
        action = self.agent.step_inference(obs)

        while True:
            obs, reward, done, _ = self.env.step(action)
            reward_episode += reward
            action = self.agent.step_inference(obs)

            if done:
                break

        return reward_episode

    def collect_data(self, n_data_collect: int) -> Tuple[List[Transition], float]:
        self.agent.actor.cpu()
        data = []
        rewards = []
        for _ in range(n_data_collect):
            r, d = self.run_episode_train()
            data.extend(d)
            rewards.append(r)

        avg_reward = sum(rewards) / len(rewards)

        return data, avg_reward
