# -*- coding: utf-8 -*-
#
# agents/q_learning_mountain_car_agent.py
#
# Q-Learning implementation for mountain car problem

import gym
import numpy as np


class QLearningMountainCarAgent:
    def __init__(self):
        self.observation_dim = 2
        self.action_dim = 3
        self.feature_dim = self.observation_dim * self.action_dim + 1

        self.num_episode = 3000
        self.num_max_step = 500

        # model design
        self.gamma = 0.99
        self.weight = (np.random.rand(self.feature_dim) - 0.5) / 50

    def calc_feature(self, observation, action):
        bias = np.array([0.1])
        feature = np.zeros(self.observation_dim * self.action_dim)
        feature[self.observation_dim*action:self.observation_dim*(action+1)] = observation
        feature = np.concatenate([bias, feature])
        return feature

    def select_action(self, observation, eps):
        action_values = [self._q_hat(self.calc_feature(observation, action)) for action in range(self.action_dim)]
        action = self._epsilon_greedy(action_values, eps)
        return action

    def train(self, env, ini_eps=0.3, ini_alpha=0.1):
        for i_episode in range(self.num_episode):
            if i_episode % 50 == 0:
                self.test(env)
            observation = env.reset()
            # action = self.select_action(observation, ini_eps)
            for i_step in range(self.num_max_step):
                eps = self._calc_epsilon(i_episode, ini_eps)
                action = self.select_action(observation, eps)
                observation_prime, reward, done, info = env.step(action)

                feature = self.calc_feature(observation, action)

                alpha = self._calc_alpha(i_episode, ini_alpha)
                action_values_prime = [self._q_hat(self.calc_feature(observation_prime, action))
                                       for action in range(self.action_dim)]
                action_value = self._q_hat(self.calc_feature(observation, action))
                td_error = reward + self.gamma * max(action_values_prime) - action_value
                self.weight += alpha * td_error * feature

                if observation_prime[0] >= 0.5:
                    print(f'success! episode: {i_episode}, step: {i_step}, epsilon: {eps}, learning rate: {alpha}')
                    break

                observation = observation_prime

    def demonstrate(self, env):
        observation = env.reset()
        for i in range(self.num_max_step):
            env.render()
            action = self.select_action(observation, 0)
            observation, reward, done, info = env.step(action)
            if observation[0] >= 0.5:
                print(f'success at step {i}')
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
    agent = QLearningMountainCarAgent()
    agent.train(env)
    agent.test(env)
    agent.demonstrate(env)
