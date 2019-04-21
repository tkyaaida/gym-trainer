#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from dataclasses import dataclass
import random
from typing import List
import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    next_obs: np.ndarray
    reward: float
    done: bool


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, obs: np.ndarray, action: int, next_obs: np.ndarray, reward: float, done: bool) -> None:
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(obs, action, next_obs, reward, done)
        self.position = (self.position + 1) % self.capacity

    def bulk_push(self, data: List[Transition]) -> None:
        for dp in data:
            self.push(dp.obs, dp.action, dp.next_obs, dp.reward, dp.done)

    def sample(self, batch_size: int) -> List[Transition]:
        """sample batch"""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

