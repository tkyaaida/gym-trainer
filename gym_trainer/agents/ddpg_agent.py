#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# agents/ddpg_agent.py
#
# DDPG agent

from typing import List, Tuple
import copy
from logging import getLogger
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from gym_trainer.helpers.replay_memory import Transition
from gym_trainer.helpers.logger import Logger


logger = getLogger(__name__)


class Actor(nn.Module):
    def __init__(self, dim_obs: int, dim_action: int, dim_hidden: int) -> None:
        super(Actor, self).__init__()

        # layer
        self.l1 = nn.Linear(dim_obs, dim_hidden, bias=False)
        self.l2 = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.l3 = nn.Linear(dim_hidden, dim_action, bias=False)

    def forward(self, x) -> Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, dim_obs: int, dim_action: int, dim_hidden: int) -> None:
        super(Critic, self).__init__()

        # layer
        self.l1 = nn.Linear(dim_obs + dim_action, dim_hidden, bias=False)
        self.l2 = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.l3 = nn.Linear(dim_hidden, dim_action, bias=False)

    def forward(self, x_obs: Tensor, x_action: Tensor) -> Tensor:
        """

        Args:
            x_obs: (batch_size, dim_obs)
            x_action: (batch_size, dim_action)

        Returns:

        """
        x = torch.cat([x_obs, x_action], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


class QUNoise:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(size, dtype=np.float32)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.rand(self.size).astype(np.float32)
        self.state = x + dx
        return self.state


class DDPGAgent:
    def __init__(self, dim_obs: int, dim_action: int, dim_hidden: int, gamma: float, tau: float, device: str,
                 use_polyak: bool = False, target_update: int = 10, lr_actor: float = 0.01,
                 lr_critic: float = 0.1):
        # model
        self.actor = Actor(dim_obs, dim_action, dim_hidden).to(device)
        self.actor_target = Actor(dim_obs, dim_action, dim_hidden).to(device)
        self.critic = Critic(dim_obs, dim_action, dim_hidden).to(device)
        self.critic_target = Critic(dim_obs, dim_action, dim_hidden).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.eval()
        self.critic_target.eval()

        # optimizer
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # noise in exploration policy
        self.noise = QUNoise(dim_action)

        # other params
        self.dim_obs = dim_obs
        self.dim_action = dim_action

        self.gamma = gamma
        self.tau = tau

        self.use_polyak = use_polyak
        self.optimize_count = 0
        self.target_update = target_update

        self.device = device

    def reset(self):
        self.noise.reset()

    def step(self, obs: np.ndarray):
        """assumes actor is in CPU and eval mode"""
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        assert obs.shape == (1, self.dim_obs)

        with torch.no_grad():
            action = self.actor(obs).numpy()[0, :] + self.noise.sample()
            assert action.shape == (self.dim_action, )
            action = np.clip(action, -1, 1)
        return action

    def step_inference(self, obs):
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        assert obs.shape == (1, self.dim_obs)

        with torch.no_grad():
            action = self.actor(obs).numpy()[0, :]
            return action

    def to_cpu(self):
        """move all model to cpu"""
        self.actor.to('cpu')
        self.actor_target.to('cpu')
        self.critic.to('cpu')
        self.critic_target.to('cpu')

    def to_gpu(self):
        """move all model to GPU"""
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    def optimize(self, batch):
        self.optimize_count += 1
        state, action, state_prime, reward, done = self.transform_batch(batch)

        # -- update critic --
        # compute action_prime according to Double Q-learning manner
        with torch.no_grad():
            action_prime = self.actor_target(state_prime)
            q_target = self.critic_target(state_prime, action_prime)
            q_target = reward + self.gamma * q_target * (1 - done)  # if done, q_target is 0

        q_values = self.critic(state, action)
        loss = F.smooth_l1_loss(q_values, q_target)

        # update
        self.optim_critic.zero_grad()
        loss.backward()
        for param in self.critic.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim_critic.step()
        logger.info(f'Critic loss: {loss.data.to("cpu")}')

        # -- update actor --
        action_pred = self.actor(state)
        loss = -self.critic(state, action_pred).mean()  # critic(state, actor(state))

        # update
        self.optim_actor.zero_grad()
        loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim_actor.step()
        logger.info(f'Actor loss: {loss.data.to("cpu")}')

        # upadte target
        if self.use_polyak:
            for param, param_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                param_target.data.copy_(self.tau * param_target.data + (1-self.tau) * param.data)
            for param, param_target in zip(self.actor.parameters(), self.actor_target.parameters()):
                param_target.data.copy_(self.tau * param_target.data + (1-self.tau) * param.data)
        elif self.optimize_count % self.target_update == 0:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.actor_target.load_state_dict(self.actor.state_dict())

    def transform_batch(self, batch: List[Transition]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """convert batch to array"""
        state = []
        action = []
        state_prime = []
        reward = []
        done = []

        for x in batch:
            state.append(x.obs)
            action.append(x.action)
            state_prime.append(x.next_obs)
            reward.append(x.reward)
            done.append(x.done)

        state = torch.Tensor(state)
        action = torch.Tensor(action)
        state_prime = torch.Tensor(state_prime)
        reward = torch.Tensor(reward)
        done = torch.Tensor(done)

        if self.device != torch.device('cpu'):
            state = state.to(self.device, dtype=torch.float)
            action = action.to(self.device, dtype=torch.float)
            state_prime = state_prime.to(self.device, dtype=torch.float)
            reward = reward.to(self.device, dtype=torch.float)
            done = done.to(self.device, dtype=torch.float)
        else:
            state = state.to(torch.float)
            action = action.to(torch.float)
            state_prime = state_prime.to(torch.float)
            reward = reward.to(torch.float)
            done = done.to(torch.float)

        return state, action, state_prime, reward, done
