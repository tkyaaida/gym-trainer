#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import argparse
from argparse import Namespace
from datetime import datetime
from pathlib import Path
import gym
import torch
from gym_trainer.agents.dqn_agent import DQNAgent
from gym_trainer.helpers.replay_memory import ReplayMemory
from gym_trainer.interactions.dqn_interaction import DQNInteraction
from gym_trainer.helpers.logger import Logger, Module
from gym_trainer.helpers.config import Config


def get_args() -> Namespace:
    parser = argparse.ArgumentParser('DQN agent')
    parser.add_argument('--n_epoch', '-ne', type=int, default=1000,
                        help='number of epoch')
    parser.add_argument('--n_data_collect', '-ndc', type=int, default=128,
                        help='number of data points to collect in each epoch')
    parser.add_argument('--n_batch', '-nb', type=int, default=128,
                        help='batch size')
    parser.add_argument('--use_polyak_average', '-upa', action='store_true',
                        help='use polyak averaging')
    parser.add_argument('--tau', '-t', type=float, default=0.999,
                        help='ratio of using weight of previous target network')
    parser.add_argument('--n_update_target_iter', '-nuti', type=int, default=1,
                        help='number of iterations to update target network parameter')
    parser.add_argument('--n_batch_sample', '-nbs', type=int, default=1,
                        help='number of times to sample batches')
    parser.add_argument('--n_memory_capacity', '-nmc', type=int, default=10000,
                        help='capacity of replay memory')
    parser.add_argument('--gamma', '-g', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--eps_start', '-es', type=float, default=0.9,
                        help='epsilon at start')
    parser.add_argument('--eps_end', '-ee', type=float, default=0.05,
                        help='epsilon at the end')
    parser.add_argument('--eps_decay', '-ed', type=int, default=0.99,
                        help='rate of decay')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
    # evaluation
    parser.add_argument('--n_eval_epoch', '-nee', type=int, default=50,
                        help='number of epochs per evaluation')
    parser.add_argument('--n_output_epoch', '-noe', type=int, default=50)
    # general
    parser.add_argument('--device', '-d', default='cuda:0',
                        help='"cuda:0" for GPU 0 or "cpu" for cpu')
    args = parser.parse_args()
    return args


def setup_output_directory(args: Namespace):
    out_dir = Config.EXP_OUT_DIR / 'MountainCar' / 'DQN'  # type: Path
    if args.out:
        out_dir = out_dir / args.out
    else:
        out_dir = out_dir / f'exp-{datetime.now().strftime("%Y%m%d%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def setup_logger(out_dir: Path) -> Logger:
    name = 'train_dqn'
    fname = str(out_dir / f'{name}.log')
    modules = [Module('gym_trainer.agents.dqn_agent')]
    logger = Logger(name, fname, modules)
    return logger


def setup_interaction(args: Namespace) -> DQNInteraction:
    device = torch.device(args.device)

    env = gym.make('MountainCar-v0')
    agent = DQNAgent(2, 3, 32, args.eps_start, args.eps_end, args.eps_decay, args.gamma, args.tau, device,
                     args.use_polyak_average, lr=args.learning_rate)
    dqn_intr = DQNInteraction(agent, env)
    return dqn_intr


def run_epoch(args: Namespace, dqn_intr: DQNInteraction, memory: ReplayMemory,
              i_epoch: int, logger: Logger):
    device = torch.device(args.device)

    # collect data
    dqn_intr.agent.policy_net.cpu()

    rewards = []
    data = []
    for _ in range(args.n_data_collect):
        r, d = dqn_intr.run_episode_train(i_epoch)
        rewards.append(r)
        data.extend(d)

    avg_reward = sum(rewards) / len(rewards)
    logger.info(f'Epoch: {i_epoch} avg reward: {avg_reward}')

    # for each batch, compute y and update
    dqn_intr.agent.policy_net.to(device)
    for _ in range(args.n_batch_sample):
        batch = memory.sample(args.n_batch)
        dqn_intr.agent.optimize(batch)

    # evaluate
    if i_epoch % args.n_eval_epoch == 0:
        avg_reward = evaluate(dqn_intr)
        logger.info(f'[EVALUATION]: avg reward: {avg_reward}')

    # save model
    if i_epoch % args.n_output_epoch == 0:
        pass


def evaluate(dqn_intr: DQNInteraction) -> float:
    dqn_intr.agent.policy_net.cpu()
    rewards = []
    for _ in range(10):
        r = dqn_intr.run_episode_eval()
        rewards.append(r)

    avg_reward = sum(rewards) / len(rewards)
    return avg_reward


def main():
    args = get_args()

    out_dir = setup_output_directory(args)
    logger = setup_logger(out_dir)

    dqn_intr = setup_interaction(args)
    memory = ReplayMemory(args.n_memory_capacity)

    logger.info('Start Training')
    for i_epoch in range(args.n_epoch):
        run_epoch(args, dqn_intr, memory, i_epoch, logger)


if __name__ == '__main__':
    main()
