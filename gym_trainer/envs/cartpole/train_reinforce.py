#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import argparse
import pickle
import gym
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.functions as F
from chainer.training import Trainer
from chainer.optimizers import MomentumSGD
from chainer.backends import cuda
from gym_trainer.agents.reinforce_agent import ReinforceAgent
from gym_trainer.helpers.logger import Logger


logger = Logger(__file__)


def convert(samples, device):
    """convert samples (list of trajectories) into ndarray and send to device"""
    obs = []
    actions = []
    rewards = []

    for sample in samples:
        for transition in sample:
            obs.append(transition['obs'])
            actions.append(transition['action'])
            rewards.append(transition['reward'])

    n = len(samples)
    t = len(samples[0])

    obs = np.vstack(obs)
    obs = obs.astype(np.float32)
    assert obs.shape[0] == n * t
    actions = np.array(actions, dtype=np.int32)
    assert actions.shape == (n * t, )
    rewards = np.array(rewards, dtype=np.float32).reshape((n, t))

    if device >= 0:
        cuda.to_gpu(obs, device=device)
        cuda.to_gpu(actions, device=device)
        cuda.to_gpu(rewards, device=device)

    return obs, actions, rewards


def main():
    parser = argparse.ArgumentParser(description='CartPole-v0 with REINFORCE agent')
    parser.add_argument('--n_epoch', '-ne', type=int, default=2000,
                        help='number of epoch')
    parser.add_argument('--n_step', '-ns', type=int, default=200,
                        help='maximum number of steps for each trajectory')
    parser.add_argument('--n_rollout', '-nr', type=int, default=20,
                        help='number of trajectories to generate in each iteration')
    parser.add_argument('--lr', '-lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--weight_decay_rate', '-wdr', type=float, default=0.00001,
                        help='L2 reguralization')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value means CPU)')
    args = parser.parse_args()

    # prepare env and agent
    env = gym.make('CartPole-v0')
    agent = ReinforceAgent(4, 2)
    if args.gpu >= 0:
        agent.policy.to_gpu(args.gpu)

    optimizer = MomentumSGD(lr=args.lr)
    optimizer.setup(agent.policy)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(args.weight_decay_rate))

    # training loop
    losses = []
    total_rewards = []
    for i_epoch in range(args.n_epoch):
        # generate samples
        samples = []
        for i_rollout in range(args.n_rollout):
            trajectory = []
            obs = env.reset()
            action = agent.reset_inference(obs)
            done = False

            for i_step in range(args.n_step):
                if done is False:
                    next_obs, reward, done, info = env.step(action)
                    next_action = agent.step_inference(next_obs)
                    trajectory.append({
                        'obs': obs,
                        'action': action,
                        'reward': reward,
                    })
                    obs = next_obs
                    action = next_action

                else:
                    # padding
                    trajectory.append({
                        'obs': obs,
                        'action': -1,  # ignore label
                        'reward': 0,
                    })

            samples.append(trajectory)

        tmp_rewards = []
        for sample in samples:
            reward = 0
            for trajectory in sample:
                reward += trajectory['reward']
            tmp_rewards.append(reward)
        avg_reward = sum(tmp_rewards) / len(tmp_rewards)
        logger.info(f'epoch: {i_epoch}, avg. reward: {avg_reward}')
        total_rewards.append(avg_reward)

        # evaluate gradient and optimize parameter
        obs, actions, rewards = convert(samples, args.gpu)
        ce = F.softmax_cross_entropy(agent.policy.forward(obs), actions, reduce='no')
        assert ce.shape == (args.n_rollout * args.n_step, )
        reward_to_go = agent.calc_reward_to_go(rewards)
        loss = ce * reward_to_go
        loss = loss.reshape((args.n_rollout, args.n_step))
        loss = F.mean(F.sum(loss, axis=1))
        agent.policy.cleargrads()
        loss.backward()
        optimizer.update()

        logger.info(f'epoch: {i_epoch}, loss: {loss.array}')
        losses.append(loss.array)

        # evaluate once in a while
        # if i_epoch % 30 == 0:
        #     rewards = []
        #     for _ in range(10):
        #         obs = env.reset()
        #         action = agent.reset_inference(obs)
        #         reward_episode = 0
        #         while True:
        #             obs, reward, done, _ = env.step(action)
        #             reward_episode += reward
        #             action = agent.step_inference(obs)
        #             if done:
        #                 rewards.append(reward_episode)
        #                 break
        #     avg_reward = sum(rewards) / len(rewards)
        #     logger.info('---------------------------------------------')
        #     logger.info(f'avg reward: {avg_reward}')
        #     logger.info('---------------------------------------------')
        #     total_rewards.append(avg_reward)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(np.arange(len(losses)), np.array(losses))
    ax = fig.add_subplot(212)
    ax.plot(np.arange(len(total_rewards)), total_rewards)
    fig.savefig('plot.png')


if __name__ == '__main__':
    main()
