#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from gym_trainer.agents.table_q_learning_agent import TableQLearningAgent


class TableQInteraction:
    def __init__(self, agent: TableQLearningAgent, env):
        self.agent = agent
        self.env = env

    def run_episode_train(self, i_episode) -> float:
        pass

    def run_episode_eval(self) -> float:
        pass
