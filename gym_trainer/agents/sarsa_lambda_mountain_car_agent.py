# -*- coding: utf-8 -*-
#
# agents/sarsa_lambda_mountain_car_agent.py
#
# sarsa(λ) implementation for mounrain car problem

import gym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time


class SarsaLambdaMountainCarAgent:
    def __init__(self):
        self.observation_dim = 2
        self.action_dim = 3
        self.feature_dim = self.observation_dim * self.action_dim + 1
        self.num_max_step = 500  # maximum step per episode
        self.num_episode = 3000

        # model design
        self.decay_rate = 0.75  # lambda: the rate at which decreases return value
        self.gamma = 0.99  # discount factor: the rate at which discounted
        self.weight = (np.random.rand(self.feature_dim) - 0.5) / 50  # initial weight [-0.01, 0.01)

    def calc_feature(self, observation, action):
        """given observation

        Args:
            observation:
            action:

        Returns:

        """
        bias = np.array([1])
        feature = np.zeros(self.observation_dim * self.action_dim)
        feature[self.observation_dim*action:self.observation_dim*(action+1)] = observation
        feature = np.concatenate([bias, feature])
        return feature

    def select_action(self, observation, eps):
        """select action based on action values with ε = eps

        Args:
            observation:
            eps:

        Returns:

        """
        action_values = [self._q_hat(self.calc_feature(observation, action)) for action in range(self.action_dim)]
        action = self._epsilon_greedy(action_values, eps)
        return action

    def train(self, env, ini_eps=0.3, ini_alpha=0.1):
        success_counter = 0
        epsilons = []
        alphas = []
        for i_episode in range(self.num_episode):
            if i_episode % 100 == 0:
                print(f'episode num: {i_episode}')
                print(self.weight)

            observation = env.reset()
            action = self.select_action(observation, ini_eps)
            feature = self.calc_feature(observation, action)
            z = np.zeros(self.feature_dim)

            for i_step in range(self.num_max_step):
                observation_prime, reward, done, info = env.step(action)
                eps = self._calc_epsilon(i_episode, ini_eps)
                action_prime = self.select_action(observation_prime, eps)
                feature_prime = self.calc_feature(observation_prime, action_prime)
                q = self._q_hat(feature)
                q_prime = self._q_hat(feature_prime)
                td_error = reward + self.gamma * q_prime - q
                alpha = self._calc_alpha(i_episode, ini_alpha)
                z = self.gamma * self.decay_rate * z + feature
                self.weight += alpha * td_error * z

                if observation_prime[0] >= 0.5:
                    print(f'success! episode: {i_episode}, step: {i_step}, epsilon: {eps}, learning rate: {alpha}')
                    success_counter += 1
                    break

                feature = feature_prime
                action = action_prime

            if i_episode % 50 == 0 and self.test(env):
                print('enough training')
                break

    def demonstrate(self, env):
        observation = env.reset()
        for i in range(self.num_max_step):
            env.render()
            action = self.select_action(observation, 0)
            observation, reward, done, info = env.step(action)
            if observation[0] >= 0.5:
                print(f'success: step {i}')
                break
        env.close()

    def test(self, env, test_num=10):
        success_counter = 0
        rewards = []
        for j in range(test_num):
            observation = env.reset()
            rewards.append(0)
            for i in range(self.num_max_step):
                action = self.select_action(observation, 0)
                observation, reward, done, info = env.step(action)
                rewards[j] += reward
                if observation[0] >= 0.5:
                    success_counter += 1
                    break
        print(f'[TEST RESULT] # of test: {test_num}, # of success: {success_counter}, rewards: {rewards}')
        if success_counter / test_num >= 0.8:
            return True
        else:
            return False

    def _q_hat(self, feature):
        return np.dot(self.weight, feature)

    def _epsilon_greedy(self, action_values, eps):
        is_greedy = np.random.binomial(n=1, p=1-eps)
        if is_greedy:
            return np.argmax(action_values)
        else:
            return np.random.randint(len(action_values))

    def _calc_epsilon(self, i_episode, ini_eps):
        eps = max(ini_eps - i_episode / 2000, 0.03)
        return eps

    def _calc_alpha(self, i_episode, ini_alpha):
        # alpha = max(0.5 - i_episode / self.num_episode, 0.001)
        alpha = ini_alpha / (i_episode / 1000 + 1)
        return alpha


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    agent = SarsaLambdaMountainCarAgent()
    agent.train(env)
    agent.demonstrate(env)
