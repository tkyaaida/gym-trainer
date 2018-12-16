#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# agents/dqn_agent.py
#
# DQN agent

from typing import List, Tuple
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from gym_trainer.helpers.replay_memory import Transition
from gym_trainer.helpers.utils import get_decayed_param


class DQN(nn.Module):
    def __init__(self, dim_obs: int, dim_action: int, dim_hidden: int):
        super(DQN, self).__init__()
        self.dim_obs = dim_obs

        # layer
        self.l1 = nn.Linear(dim_obs, dim_hidden, bias=False)
        self.l2 = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.l3 = nn.Linear(dim_hidden, dim_action, bias=False)

    def forward(self, x):
        """

        Args:
            x : shape: (batchsize, dim_obs)

        Returns:

        """
        assert x.shape[1] == self.dim_obs
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        y = self.l3(h)
        return y


class DQNAgent:
    def __init__(self, dim_obs, dim_action, dim_hidden, eps_start, eps_end, eps_decay, gamma, tau, device,
                 use_polyak=True, lr=0.1):
        self.policy_net = DQN(dim_obs, dim_action, dim_hidden).to(device)
        self.target_net = DQN(dim_obs, dim_action, dim_hidden).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.policy_net.to(device)
        self.target_net.to(device)

        self.dim_action = dim_action

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.tau = tau

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=lr)
        self.device = device

    def step(self, obs: np.ndarray, i_epoch: int) -> int:
        """select action"""
        eps = get_decayed_param(self.eps_start, self.eps_end, self.eps_decay, i_epoch)
        is_greedy = np.random.binomial(n=1, p=1-eps)
        if is_greedy:
            obs = torch.from_numpy(obs).float()
            obs.unsqueeze_(0)  # add batch dimension
            with torch.no_grad():
                actions = self.policy_net(obs).squeeze(0)
                return int(torch.argmax(actions))
        else:
            return np.random.randint(self.dim_action)

    def step_inference(self, obs: np.ndarray) -> int:
        obs = torch.from_numpy(obs).float()
        obs.unsqueeze_(0)  # add batch dimension
        with torch.no_grad():
            actions = self.policy_net(obs).squeeze(0)
            return int(torch.argmax(actions))

    def optimize(self, batch: List[Transition]) -> None:
        batch_size = len(batch)
        # convert batch
        state, action, state_prime, reward = self.transform_batch(batch)

        # compute loss
        with torch.no_grad():
            self.policy_net.eval()
            action_prime = torch.argmax(self.policy_net(state_prime), dim=1)
            expected_action_values = reward + \
                                     self.gamma * self.target_net(state_prime)[torch.arange(batch_size), action_prime]

        self.policy_net.train()
        action_values = self.policy_net(state)[torch.arange(batch_size), action]
        loss = F.smooth_l1_loss(action_values, expected_action_values)

        # update
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Polyak averaging
        for param, param_target in zip(self.policy_net.parameters(), self.target_net.parameters()):
            param_target.data.copy_(self.tau * param_target.data + (1-self.tau) * param.data)

    def transform_batch(self, batch: List[Transition]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """convert batch to array"""
        state = []
        action = []
        state_prime = []
        reward = []

        for x in batch:
            state.append(x.obs)
            action.append(x.action)
            state_prime.append(x.next_obs)
            reward.append(x.reward)

        state = torch.Tensor(state)
        action = torch.Tensor(action)
        state_prime = torch.Tensor(state_prime)
        reward = torch.Tensor(reward)

        if self.device != torch.device('cpu'):
            state = state.to(self.device, dtype=torch.float)
            action = action.to(self.device, dtype=torch.long)
            state_prime = state_prime.to(self.device, dtype=torch.float)
            reward = reward.to(self.device, dtype=torch.float)
        else:
            state = state.to(torch.float)
            action = action.to(torch.long)
            state_prime = state_prime.to(torch.float)
            reward = reward.to(torch.float)

        return state, action, state_prime, reward
