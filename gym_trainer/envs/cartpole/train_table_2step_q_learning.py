#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import os
import argparse
import pickle
import gym
from gym_trainer.agents.table_2step_q_learning_agent import Table2StepQLearningAgent


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
    parser.add_argument('--out', '-o', default='result',
                        help='name of output directory')
    args = parser.parse_args()

    os.mkdir(args.out)

    # prepare env and agent
    env = gym.make('CartPole-v0')
    obs_min = [-2.4, -3.0, -0.5, -2.0]
    obs_max = [2.4, 3.0, 0.5, 2.0]
    agent = Table2StepQLearningAgent(4, 2, obs_min, obs_max, args.n_disc)

    # training loop
    for i_episode in range(args.n_episode):
        if i_episode % 100 == 0:
            print(f'{i_episode/args.n_episode*100} % done')
        obs = env.reset()
        action = agent.reset(obs, i_episode)
        obs_next, reward, done, info = env.step(action)
        action_next = agent.step_inference(obs_next)

        if done:
            continue

        for i_step in range(args.n_step):
            obs_next_next, reward_next, done, info = env.step(action_next)
            action_next_next = agent.step(obs, action, obs_next_next, reward, reward_next, i_episode)

            obs = obs_next
            obs_next = obs_next_next
            action = action_next
            action_next = action_next_next
            reward = reward_next

            if done:
                break

    # save agent
    with open(args.out + '/table_2step_q_learning_agent.dump', 'wb') as f:
        pickle.dump(agent, f)


if __name__ == '__main__':
    main()

