#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import argparse
import multiprocessing as multi
from copy import copy
import gym
import numpy as np
import matplotlib.pyplot as plt
from chainer.optimizers import Adam
from gym_trainer.agents.reinforce_agent_chainer import ReinforceAgent
from gym_trainer.helpers.logger import Logger


logger = Logger(__file__)


def run(agent: ReinforceAgent, env: gym.Env, n_step: int):
    """run policy to get trajectory"""
    trajectory = []
    total_reward = 0
    obs = env.reset()
    action = agent.reset_inference(obs)
    done = False

    for _ in range(n_step):
        if done is False:
            next_obs, reward, done, info = env.step(action)
            next_action = agent.step_inference(next_obs)
            total_reward += reward
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
    return trajectory, total_reward


def main():
    parser = argparse.ArgumentParser(description='CartPole-v0 with REINFORCE agent')
    parser.add_argument('--n_epoch', '-ne', type=int, default=200,
                        help='number of epoch')
    parser.add_argument('--n_step', '-ns', type=int, default=200,
                        help='maximum number of steps for each trajectory')
    parser.add_argument('--n_rollout', '-nr', type=int, default=20,
                        help='number of trajectories to generate in each iteration')
    parser.add_argument('--lr', '-lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay_rate', '-wdr', type=float, default=0.00001,
                        help='L2 reguralization')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value means CPU)')
    parser.add_argument('--n_cpu', '-nc', type=int, default=multi.cpu_count(),
                        help='number of CPUs to use')
    args = parser.parse_args()

    # prepare env and agent
    env = gym.make('CartPole-v0')
    optimizer = Adam(alpha=args.lr)
    agent = ReinforceAgent(4, 2, optimizer, args.gpu)
    if args.gpu >= 0:
        agent.policy.to_gpu(args.gpu)

    # training loop
    loss_history = []
    reward_history = []
    logger.info('Start training loop')
    for i_epoch in range(args.n_epoch):
        # generate samples
        with multi.Pool(args.n_cpu) as p:
            samples = p.starmap(run, [(copy(agent), copy(env), args.n_step) for _ in range(args.n_rollout)])

        trajectories, reward = zip(*samples)  # list of tuples to two separate tuples

        avg_reward = sum(reward) / len(reward)
        logger.info(f'epoch: {i_epoch}, avg. reward: {avg_reward}')
        reward_history.append(avg_reward)

        # evaluate gradient and optimize parameter
        logger.info('Evaluate gradient and optimize params')
        loss = agent.step(trajectories)
        logger.info(f'epoch: {i_epoch}, loss: {loss}')
        loss_history.append(loss)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(np.arange(len(loss_history)), np.array(loss_history))
    ax = fig.add_subplot(212)
    ax.plot(np.arange(len(reward_history)), reward_history)
    fig.savefig('plot.png')


if __name__ == '__main__':
    main()
