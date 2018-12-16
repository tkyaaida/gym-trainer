# -*- coding: utf-8 -*-
#
# helpers/utils.py
#
# utility functions


import numpy as np


def epsilon_greedy(epsilon, action_values):
    """ε-greedyにaction_valuesに対応するactionのindexを返す

    Args:
        epsilon (float): ε(=行動をランダムに選択する確率. 0以上1以下)
        action_values (list): 行動価値

    Returns:
        int: 次のactionのindex
    """
    assert 0 <= epsilon <= 1
    is_greedy = np.random.binomial(n=1, p=1-epsilon)
    if is_greedy:
        return np.argmax(action_values)
    else:
        return np.random.randint(len(action_values))


def get_decayed_param(initial_value, final_value, decay_rate, i_episode):
    """get decayed parameter at episode i_episode

    Args:
        initial_value (float): initial value of the parameter
        final_value (float): final value
        decay_rate (float): rate at which parameter value is decayed.
        i_episode (int): number of episode

    Returns:
        float: parameter value at episode i_episode
    """
    param = initial_value * (decay_rate ** i_episode) + final_value
    return param
