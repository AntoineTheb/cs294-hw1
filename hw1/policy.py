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

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label = 'Val loss')
    plt.legend()
    plt.ylim([0, 1])



class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
           print('')
        print('.', end='')

def build_model(train_data, train_label):
    model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
    input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(train_label.shape[1])
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
    optimizer=optimizer,
    metrics=['mae'])
    return model


def train_model(model, train_data, train_label, test_data, test_label):
    EPOCHS = 500

    # Store training stats
    history = model.fit(train_data, train_label, epochs=EPOCHS,
            validation_split=0.2, verbose=0,
            callbacks=[PrintDot()])

    plot_history(history)

    [loss, mae] = model.evaluate(test_data, test_label, verbose=0)
    print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

    return model

def play(policy, envname, max_timesteps, num_rollouts):
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
        while True:
            env.render()
            action = policy.predict(obs[None,:])[0]
            #obs, r, done, _ = env.step(env.action_space.sample())
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                # print('Done after {} steps'.format(steps))
                # print('Total reward', totalr)
                break

        returns.append(totalr)

    # print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

def get_data(observations, actions):
    test_data_size = int(len(observations) / 10)
    train_data = np.array(observations[test_data_size:])
    test_data = np.array(observations[:test_data_size])
    train_label = np.array(actions[test_data_size:])
    test_label = np.array(actions[:test_data_size])

    return train_data, test_data, train_label, test_label

