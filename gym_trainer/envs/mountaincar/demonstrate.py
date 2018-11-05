#!/usr/bin/envs python
# -*- coding: utf-8 -*-
#

import argparse
import pickle
import gym


def main():
    parser = argparse.ArgumentParser(description='demo script for MountainCar task')
    parser.add_argument('--number', '-n', type=int, default=10,
                        help='number of iteration')
    parser.add_argument('--agent', '-a', type=str,
                        help='path to the agent file')
    parser.add_argument('--step', '-s', type=int, default=250,
                        help='maximum step')
    args = parser.parse_args()

    env = gym.make('MountainCar-v0')
    with open(args.agent, 'rb') as f:
        agent = pickle.load(f)

    for _ in range(args.number):
        observation = env.reset()
        action = agent.reset(observation)
        for i_step in range(args.step):
            env.render()
            observation, reward, done, info = env.step(action)

            if done:
                print(i_step)
                break

            action = agent.step_inference(observation)

    env.close()


if __name__ == '__main__':
    main()
