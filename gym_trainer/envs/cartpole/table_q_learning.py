#!/usr/bin/envs python
# -*- coding: utf-8 -*-
#

import argparse
import pickle
import gym
from gym_trainer.agents.table_q_learning_agent import TableQLearningAgent


def main():
    parser = argparse.ArgumentParser(description='CartPole with table Q-learning agent')
    parser.add_argument('--n_episode', '-e', type=int, default=1000,
                        help='number of episodes to train')
    parser.add_argument('--n_step', '-s', type=int, default=200,
                        help='number of steps at each episode')
    parser.add_argument('--n_disc', '-n', type=int, default=10,
                        help='number of discretization')
    args = parser.parse_args()

    # prepare envs and agent
    env = gym.make('CartPole-v0')
    obs_min = [-2.4, -3.0, -0.5, -2.0]
    obs_max = [2.4, 3.0, 0.5, 2.0]
    agent = TableQLearningAgent(4, 2, obs_min, obs_max, args.n_disc)

    # training loop
    for i_episode in range(args.n_episode):
        obs = env.reset()
        action = agent.reset(obs)

        if i_episode % 100 == 0:
            print(agent.q_values)

        for i_step in range(args.n_step):
            obs, reward, done, info = env.step(action)

            if done and i_step < 195:
                reward = -195

            action = agent.step(obs, reward, action)

            if done:
                break

    # save agent
    with open('table_q_learning_agent.dump', 'wb') as f:
        pickle.dump(agent, f)


if __name__ == '__main__':
    main()
