#!/usr/bin/envs python
# -*- coding: utf-8 -*-
#
# CartPole task with Q-learning agent
#
# Environment description: https://github.com/openai/gym/wiki/CartPole-v0
#

import argparse
import pickle
import gym
from gym_trainer.agents.linear_q_learning_agent import LinearQLearningAgent


def main():
    # prepare argument parser
    parser = argparse.ArgumentParser(description='CartPole with Q-learning agent')
    parser.add_argument('--n_episode', '-e', type=int, default=1000,
                        help='number of episodes to train')
    parser.add_argument('--n_step', '-s', type=int, default=250,
                        help='number of steps at each episode')
    args = parser.parse_args()

    # prepare envs and agent
    env = gym.make('CartPole-v0')
    agent = LinearQLearningAgent(4, 2)

    # training loop
    for i_episode in range(args.n_episode):
        obs = env.reset()
        action = agent.reset(obs)

        if i_episode % 100 == 0:
            print(agent.w)

        for i_step in range(args.n_step):
            obs, reward, done, info = env.step(action)

            # modify reward
            if done and i_step < 195:
                reward = -195

            action = agent.step(obs, reward, i_episode, action)

            if done:
                break

    # save agent
    with open('agent.dump', 'wb') as f:
        pickle.dump(agent, f)


if __name__ == '__main__':
    main()
