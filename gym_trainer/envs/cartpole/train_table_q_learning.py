#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import os
import argparse
import pickle
import gym
from gym_trainer.agents.table_q_learning_agent import TableQLearningAgent


def main():
    parser = argparse.ArgumentParser(description='CartPole with table Q-learning agent')
    parser.add_argument('--n_episode', '-ne', type=int, default=1000,
                        help='number of episodes to train')
    parser.add_argument('--n_disc', '-nd', type=int, default=5,
                        help='number of discretization for state')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--gamma', '-g', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--eps_ini', '-eini', type=float, default=0.8,
                        help='initial value of epsilon')
    parser.add_argument('--eps_inf', '-einf', type=float, default=0.05,
                        help='stationary value of epsilon')
    parser.add_argument('--eps_decay_rate', '-edr', type=float, default=0.99,
                        help='decay rate of epsilon value')
    parser.add_argument('--out', '-o', default='result_table_q',
                        help='結果出力のディレクトリ名')
    args = parser.parse_args()

    os.mkdir(args.out)

    # prepare env and agent
    env = gym.make('CartPole-v0')
    obs_min = [-2.4, -3.0, -0.5, -2.0]
    obs_max = [2.4, 3.0, 0.5, 2.0]
    agent = TableQLearningAgent(4, 2, obs_min, obs_max, args.n_disc, lr=args.learning_rate, gamma=args.gamma,
                                eps_ini=args.eps_ini, eps_inf=args.eps_inf, eps_decay_rate=args.eps_decay_rate)

    # training loop
    for i_episode in range(args.n_episode):
        if i_episode % 100 == 0:
            print(f'{i_episode/args.n_episode*100} % done')
        obs = env.reset()
        action = agent.reset(obs, i_episode)

        while True:
            obs_next, reward, done, info = env.step(action)
            action = agent.step(obs, action, obs_next, reward, i_episode)
            obs = obs_next

            if done:
                break

    # save agent
    with open(args.out + '/agent.dump', 'wb') as f:
        pickle.dump(agent, f)


if __name__ == '__main__':
    main()
