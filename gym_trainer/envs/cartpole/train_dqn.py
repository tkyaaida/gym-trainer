#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import argparse
import gym
import torch
from gym_trainer.agents.dqn_agent import DQNAgent
from gym_trainer.helpers.replay_memory import ReplayMemory
from gym_trainer.helpers.logger import Logger


logger = Logger('train_dqn')


def main():
    parser = argparse.ArgumentParser('DQN agent')
    parser.add_argument('--n_epoch', '-ne', type=int, default=1000,
                        help='number of epoch')
    parser.add_argument('--n_data_collect', '-ndc', type=int, default=128,
                        help='number of data points to collect in each epoch')
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
    parser.add_argument('--n_eval_cycle', '-nec', type=int, default=10,
                        help='number of epochs per evaluation')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
    parser.add_argument('--device', '-d', default='cuda:0',
                        help='"cuda:0" for GPU 0 or "cpu" for cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    env = gym.make('CartPole-v0')
    agent = DQNAgent(4, 2, args.n_hidden, args.eps_start, args.eps_end, args.eps_decay, args.gamma, args.tau, device,
                     args.use_polyak_average, target_update=args.n_target_update, lr=args.learning_rate)
    memory = ReplayMemory(args.n_memory_capacity)

    logger.info('Start Training')
    for i_epoch in range(args.n_epoch):
        # collect data
        agent.policy_net.cpu()
        avg_reward = 0
        for _ in range(args.n_data_collect):
            obs = env.reset()
            action = agent.step(obs, i_epoch)
            reward_sum = 0
            for t in range(200):
                next_obs, reward, done, _ = env.step(action)
                reward_sum += reward

                if done:
                    if t == 199:
                        memory.push(obs, action, next_obs, 1, done)
                    else:
                        memory.push(obs, action, next_obs, -1, done)

                    avg_reward += reward_sum
                    break
                else:
                    memory.push(obs, action, next_obs, 0, done)

                obs = next_obs
                action = agent.step(obs)

        avg_reward /= args.n_data_collect
        logger.info(f'epoch: {i_epoch}, avg reward: {avg_reward}')

        # for each batch, compute y and update
        agent.policy_net.to(device)
        for _ in range(args.n_batch_sample):
            batch = memory.sample(args.n_batch)
            agent.optimize(batch)

        # evaluate
        if i_epoch % args.n_eval_cycle == 0:
            avg_reward = 0
            agent.policy_net.cpu()
            for _ in range(10):
                obs = env.reset()
                action = agent.step_inference(obs)
                reward_sum = 0
                while True:
                    obs, reward, done, _ = env.step(action)
                    reward_sum += reward
                    action = agent.step_inference(obs)

                    if done:
                        avg_reward += reward_sum
                        break

            avg_reward /= 10
            logger.info(f'[EVALUATION]: avg reward: {avg_reward}')


if __name__ == '__main__':
    main()
