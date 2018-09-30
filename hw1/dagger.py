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

def get_experts_data(expert_file, envname, render, max_timesteps, num_rollouts):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expert_file)
    print('loaded and built')
    expert_data = {}
    env = gym.make(envname)
    max_steps = max_timesteps or env.spec.timestep_limit
    returns = []
    observations = []
    actions = []

    for i in range(num_rollouts):
        # print('iter', i)
        obs = env.reset()
        totalr = 0
        steps = 0
        done = False
        while not done:
            action = policy_fn(obs[None,:])[0]
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1

        returns.append(totalr)

    # print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))


    expert_data['observations'] = np.array(observations)
    expert_data['actions'] = np.array(actions)

    #print('Training data', expert_data['observations'].shape)
    #print('Format', expert_data['observations'][0])
    #print('Training labels', expert_data['actions'].shape)
    #print('Format', expert_data['actions'][0])
    return expert_data

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

        expert_data = dagger(args.expert_policy_file,
            args.envname,
            args.render,
            args.max_timesteps,
            args.num_rollouts)

        num_test = int(expert_data['observations'].shape[0] / 10)

        model = train_model(expert_data, num_test)
        plt.show()
        play(model, args.envname, args.max_timesteps, args.num_rollouts)

if __name__ == '__main__':
    main()
