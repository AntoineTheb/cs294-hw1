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
    policy_fn = load_policy.load_policy(expert_file)
    expert_data = {}
    returns = []
    observations = []
    actions = []

    model = None
    with tf.Session(config = tf.ConfigProto(allow_soft_placement=True)):
        with tf.device('/gpu:0'):
            tf_util.initialize()

            env = gym.make(envname)
            max_steps = max_timesteps or env.spec.timestep_limit
            for i in range(num_rollouts):
                print('Iteration {} of {}'.format(i+1, num_rollouts))
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
                    if steps >= max_steps:
                        break

                returns.append(totalr)

            expert_data['observations'] = np.array(observations)
            expert_data['actions'] = np.array(actions)

            test_data_size = int(expert_data['observations'].shape[0] / 10)
            train_data = expert_data['observations'][test_data_size:]
            test_data = expert_data['observations'][:test_data_size]
            train_label = expert_data['actions'][test_data_size:]
            test_label = expert_data['actions'][:test_data_size]

            model = build_model(train_data, train_label)
            print('Training model ...')
            train_model(model, train_data, train_label, test_data, test_label, False)
            model.save('models/{}.h5'.format(envname))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--load', type=bool, default=False,
                        help='Use a pre-existing model or train a new one')


    args = parser.parse_args()
    if not args.load:
        model = get_experts_data(args.expert_policy_file,
            args.envname,
            args.render,
            args.max_timesteps,
            args.num_rollouts)

    play(args.envname, args.max_timesteps)

if __name__ == '__main__':
    main()
