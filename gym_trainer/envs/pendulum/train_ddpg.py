#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import argparse
import gym
import torch
from gym_trainer.agents.ddpg_agent import DDPGAgent
from gym_trainer.helpers.replay_memory import ReplayMemory
from gym_trainer.helpers.logger import Logger


logger = Logger('train_ddpg')


def main():
    parser = argparse.ArgumentParser('DDPG on pendulum-v0')
    parser.add_argument('--n_epoch', '-ne', type=int, default=1000,
                        help='number of epoch')
    parser.add_argument('--n_data_collect', '-ndc', type=int, default=128,
                        help='number of data points to collect in each epoch')
    parser.add_argument('--n_max_step', '-nms', type=int, default=200,
                        help='maximum number of steps')
    parser.add_argument('--n_batch', '-nb', type=int, default=128,
                        help='batch size')
    parser.add_argument('--use_polyak_average', '-upa', action='store_true', default=False,
                        help='use polyak averaging')
    parser.add_argument('--tau', '-t', type=float, default=0.999,
                        help='ratio of using weight of previous target network')
    parser.add_argument('--n_target_update', '-ntu', type=int, default=10,
                        help='number of iterations to update target network parameter')
    parser.add_argument('--n_batch_sample', '-nbs', type=int, default=1,
                        help='number of times to sample batches')
    parser.add_argument('--n_memory_capacity', '-nmc', type=int, default=10000,
                        help='capacity of replay memory')
    parser.add_argument('--n_hidden', '-nh', type=int, default=100,
                        help='number of hidden units')
    parser.add_argument('--gamma', '-g', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--eps_start', '-es', type=float, default=0.9,
                        help='epsilon at start')
    parser.add_argument('--eps_end', '-ee', type=float, default=0.05,
                        help='epsilon at the end')
    parser.add_argument('--eps_decay', '-ed', type=int, default=0.99,
                        help='rate of decay')
    parser.add_argument('--n_eval_epoch', '-nee', type=int, default=10,
                        help='number of epochs per evaluation')
    parser.add_argument('--lr_actor', '-lra', type=float, default=0.001)
    parser.add_argument('--lr_critic', '-lrc', type=float, default=0.01)
    parser.add_argument('--device', '-d', default='cuda:0',
                        help='"cuda:0" for GPU 0 or "cpu" for cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    env = gym.make('Pendulum-v0')
    agent = DDPGAgent(3, 1, args.n_hidden, args.gamma, args.tau, device, args.use_polyak_average,
                      args.n_target_update, args.lr_actor, args.lr_critic)
    memory = ReplayMemory(args.n_memory_capacity)

    logger.info('Start Training')
    for i_epoch in range(args.n_epoch):
        # collect data
        agent.to_cpu()
        avg_reward = 0

        for _ in range(args.n_data_collect):
            obs = env.reset()
            action = agent.step(obs)
            reward_sum = 0

            for t in range(args.n_max_step):
                action = action * 2.0  # scaling
                next_obs, reward, done, _ = env.step(action)
                reward_sum += reward

                memory.push(obs, action, next_obs, reward, done)

                obs = next_obs
                action = agent.step(obs)

            avg_reward += reward_sum

        avg_reward /= args.n_data_collect
        logger.info(f'epoch: {i_epoch}, avg reward: {avg_reward}')

        # Learning
        agent.to_gpu()
        for _ in range(args.n_batch_sample):
            batch = memory.sample(args.n_batch)
            agent.optimize(batch)

        # evaluate
        if i_epoch % args.n_eval_epoch == 0:
            avg_reward = 0
            agent.to_cpu()
            for _ in range(10):
                obs = env.reset()
                action = agent.step_inference(obs)
                reward_sum = 0

                for t in range(args.n_max_step):
                    action = action * 2.0
                    obs, reward, done, _ = env.step(action)
                    action = agent.step_inference(obs)
                avg_reward += reward_sum

            avg_reward /= 10
            logger.info(f'[EVALUATION] avg reward: {avg_reward}')


if __name__ == '__main__':
    main()
