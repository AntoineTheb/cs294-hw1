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

def load_model(envname):
    env = gym.make(envname)
    # Gather expert experiences
    with tf.Session(config = tf.ConfigProto(allow_soft_placement=True)):
        with tf.device('/gpu:0'):
            tf_util.initialize()
            max_steps = max_timesteps or env.spec.timestep_limit
            obs = env.reset()
            exp_action = expert(obs[None, :])[0]
            model = build_model([obs], [action])
    return model


def build_model(train_data, train_label):
    model = keras.Sequential([
        keras.layers.Dense(64, activation=keras.activations.relu,
            input_shape=(train_data.shape[1],), name='Dense1'),
        keras.layers.Dense(64, activation=keras.activations.relu, name='Dense2'),
        keras.layers.Dense(train_label.shape[1], name='Dense3')])
    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='mse',
        optimizer=optimizer,
        metrics=['accuracy'])
    return model


def train_model(model, train_data, train_label, test_data, test_label, plot):
    EPOCHS = 250

    # Store training stats
    history = model.fit(train_data, train_label, epochs=EPOCHS,
            validation_split=0.1, verbose=0, shuffle=True)

    if plot:
        plot_history(history)

    [loss, accuracy] = model.evaluate(test_data, test_label, verbose=0)
    print("Accuracy: {}".format(accuracy))
    print("Loss: {}".format(loss))

    return model

def play(envname, max_timesteps):

    with tf.Session(config = tf.ConfigProto(allow_soft_placement=True)):
        with tf.device('/gpu:0'):
            tf_util.initialize()
            policy = keras.models.load_model('models/{}.h5'.format(envname))
            env = gym.make(envname)
            max_steps = max_timesteps or env.spec.timestep_limit
            returns = []
            observations = []
            actions = []

            while(True):
                # print('iter', i)
                obs = env.reset()
                totalr = 0
                steps = 0
                done = False
                while True:
                    action = policy.predict(obs[None,:])[0]
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    env.render()
                    if steps >= max_steps:
                        break
                print(totalr)

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

