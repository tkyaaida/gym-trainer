#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import argparse
from argparse import Namespace
from pathlib import Path
from datetime import datetime
import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from gym_trainer.agents.ddpg_agent import DDPGAgent
from gym_trainer.helpers.replay_memory import ReplayMemory
from gym_trainer.interactions.ddpg_interaction import DDPGInteraction
from gym_trainer.helpers.logger import Logger, Module
from gym_trainer.helpers.config import Config


def get_args() -> Namespace:
    parser = argparse.ArgumentParser('DDPG on pendulum-v0')
    # model
    parser.add_argument('--n_hidden', '-nh', type=int, default=100,
                        help='number of hidden units')
    # training
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
    parser.add_argument('--gamma', '-g', type=float, default=0.99, help='discount factor')
    parser.add_argument('--lr_actor', '-lra', type=float, default=0.001)
    parser.add_argument('--lr_critic', '-lrc', type=float, default=0.01)
    # evaluation
    parser.add_argument('--n_eval_epoch', '-nee', type=int, default=10,
                        help='number of epochs per evaluation')
    # general
    parser.add_argument('--device', '-d', default='cuda:0',
                        help='"cuda:0" for GPU 0 or "cpu" for cpu')
    args = parser.parse_args()

    return args


def setup_output_directory() -> Path:
    out_dir = Config.EXP_OUT_DIR / 'pendulum' / 'DDPG'
    out_dir = out_dir / f'exp-{datetime.now().strftime("%Y%m%d%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def setup_logger(out_dir: Path) -> Logger:
    name = 'train_ddpg'
    fname = str(out_dir / f'{name}.log')
    modules = [Module('gym_trainer.agents.ddpg_agent')]
    logger = Logger(name, fname, modules)
    return logger


def setup_interaction(args: Namespace) -> DDPGInteraction:
    device = torch.device(args.device)

    env = gym.make('Pendulum-v0')
    agent = DDPGAgent(3, 1, np.array([-2.0]), np.array([2.0]), args.n_hidden, args.gamma, args.tau, device, args.use_polyak_average,
                      args.n_target_update, args.lr_actor, args.lr_critic)
    ddpg_intr = DDPGInteraction(agent, env)
    return ddpg_intr


def train(args: Namespace, ddpg_intr: DDPGInteraction, memory: ReplayMemory,
          logger: Logger, writer: SummaryWriter):

    for i in range(1, args.n_epoch+1):
        data, avg_reward = ddpg_intr.collect_data(args.n_data_collect)
        logger.info(f'Epoch: {i}, avg reward: {avg_reward}')
        writer.add_scalar('reward_train', avg_reward, i)
        memory.bulk_push(data)

        ddpg_intr.agent.to_gpu()
        for _ in range(args.n_batch_sample):
            batch = memory.sample(args.n_batch)
            ddpg_intr.agent.optimize(batch)

        if i % args.n_eval_epoch == 0:
            ddpg_intr.agent.to_cpu()
            r = ddpg_intr.run_episode_eval()
            logger.info(f'[Evaluation] reward: {r}')
            writer.add_scalar('reward_eval', r, i)


def main():
    args = get_args()

    out_dir = setup_output_directory()
    logger = setup_logger(out_dir)
    writer = SummaryWriter(str(out_dir))

    ddpg_intr = setup_interaction(args)
    memory = ReplayMemory(args.n_memory_capacity)

    logger.info('Start Training')
    train(args, ddpg_intr, memory, logger, writer)


if __name__ == '__main__':
    main()
