from __future__ import absolute_import, division, print_function

import os
import pickle
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import tf_util
import gym
import load_policy
import matplotlib.pyplot as plt
from policy import *

def dagger(expert_file, envname, render, max_timesteps, num_rollouts, iterations):
    policy_fn = load_policy.load_policy(expert_file)
    env = gym.make(envname)

    max_steps = max_timesteps or env.spec.timestep_limit

    expert_data = {}
    returns = []
    observations = []
    actions = []

    obs = env.reset()
    totalr = 0
    steps = 0
    done = False
    # step 0
    while not done:
        action = policy_fn(obs[None,:])[0]
        observations.append(obs)
        actions.append(action)
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1

    train_data, test_data, train_label, test_label = get_data(observations, actions)

    policy = build_model(train_data, train_label)
    for j in range(iterations):
        # step 1
        train_model(policy, train_data, train_label, test_data, test_label)
        for i in range(num_rollouts):
            obs = env.reset()
            totalr = 0
            steps = 0
            done = False
            while not done:
                action = policy.predict(obs[None,:])[0]
                exp_action = policy_fn(obs[None, :])[0]
                old_obs = obs
                # step 2
                obs, r, done, _ = env.step(action)
                # step 3
                observations.append(old_obs)
                actions.append(exp_action)

                totalr += r
                steps += 1
                if steps >= max_steps:
                    break
            # step 4
            train_data, test_data, train_label, test_label = get_data(observations, actions)

    policy = build_model(train_data, train_label)
    train_model(policy, train_data, train_label, test_data, test_label)

    return policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    with tf.Session():
        tf_util.initialize()

        model = dagger(args.expert_policy_file,
            args.envname,
            args.render,
            args.max_timesteps,
            args.num_rollouts,
            5)
        plt.show()
        play(model, args.envname, args.max_timesteps, args.num_rollouts)

if __name__ == '__main__':
    main()
