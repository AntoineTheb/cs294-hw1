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

def dagger(expert_file, envname, render, max_timesteps, iterations):
    expert = load_policy.load_policy(expert_file)
    policy = None

    totalr = 0
    steps = 0
    done = False
    expert_rollouts = 1
    observations = []
    actions = []

    env = gym.make(envname)
    # Gather expert experiences
    with tf.Session(config = tf.ConfigProto(allow_soft_placement=True)):
        with tf.device('/gpu:0'):
            tf_util.initialize()
            max_steps = max_timesteps or env.spec.timestep_limit
            obs = env.reset()

            for i in range(expert_rollouts):
                print('Expert iteration {} of {}'.format(i+1, expert_rollouts))
                steps = 0
                while not done or (not steps >= max_steps):
                    exp_action = expert(obs[None, :])[0]
                    observations.append(obs)
                    actions.append(exp_action)

                    obs, r, done, _ = env.step(exp_action)
                    steps += 1

    # DAgger
    with tf.Session(config = tf.ConfigProto(allow_soft_placement=True)):
        with tf.device('/gpu:0'):
            tf_util.initialize()
            max_steps = max_timesteps or env.spec.timestep_limit

            train_data, test_data, train_label, test_label = get_data(observations, actions)
            policy = build_model(train_data, train_label)

            for j in range(iterations):
                print('Iteration {} of {}'.format(j+1, iterations))
                # step 1
                train_data, test_data, train_label, test_label = get_data(observations, actions)
                train_model(policy, train_data, train_label, test_data, test_label, False)
                policy.save('models/{}.h5'.format(envname))


                obs = env.reset()
                totalr = 0
                steps = 0
                done = False
                # while not done:
                while not done:
                    action = policy.predict(obs[None,:])[0]
                    exp_action = expert(obs[None, :])[0]

                    # step 3
                    observations.append(obs)
                    actions.append(exp_action)

                    # step 2
                    obs, r, done, _ = env.step(action)

                    totalr += r
                    steps += 1
                    if steps >= max_steps:
                        break

                # step 4
                print('Total reward: {}\n'.format(totalr))

            print('Final training')
            train_data, test_data, train_label, test_label = get_data(observations, actions)
            train_model(policy, train_data, train_label, test_data, test_label, False)
            policy.save('models/{}.h5'.format(envname))

    return policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--iterations', type=int, default=20,
                        help='Number of iterations')
    parser.add_argument('--load', type=bool, default=False,
                        help='Use a pre-existing model or train a new one')

    args = parser.parse_args()
    if not args.load:
        dagger(args.expert_policy_file,
            args.envname,
            args.render,
            args.max_timesteps,
            args.iterations)

    play(args.envname, args.max_timesteps)

if __name__ == '__main__':
    main()
