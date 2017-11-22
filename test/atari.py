import gym
import os
import numpy as np
import scipy.ndimage as ndimage

from keras.layers import Dense, Activation, Conv2D, Dropout, Flatten, MaxPooling2D, Conv3D, MaxPooling3D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import keras.backend as K

from environments.gym import GymEnvironment
from agents import Processor
from agents.dqn import DeepQAgent, DoubleDeepQAgent
from agents.fn import ApproximateQFunction, ApproximateValueFunction
from agents.model import KerasModel
from agents.policy import QEGreedyPolicy, QERandomPolicy
from agents.mc_td import MonteCarloAgent, TemporalDifferenceAgent

LEARNING_RATE = 0.01
Q_MODEL_PATH = 'models/atari_q.h5'
V_MODEL_PATH = 'models/atari_value.h5'


class AtariScreenProcessor(Processor):
    def to_state(self, screen):
        screen = np.dot(screen, np.array([.299, .587, .114])).astype(np.uint8)
        screen = ndimage.zoom(screen, (0.4, 0.525))
        return screen.reshape((84, 84, 1))


def build_model(outputs, output_activation='linear'):
    m = Sequential()
    m.add(Conv2D(32, (8, 8), activation='relu', strides=4, padding='valid', input_shape=(84, 84, 1)))
    m.add(Conv2D(64, (4, 4), activation='relu', strides=2, padding='valid'))
    m.add(Conv2D(64, (3, 3), activation='relu', strides=1, padding='valid'))
    m.add(Flatten())
    m.add(Dense(512, activation='relu'))
    m.add(Dense(outputs, activation=output_activation))
    m.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mse')
    return m


def get_or_create_model(path, *args, **kwargs):
    if os.path.exists(path):
        return load_model(path)
    return build_model(*args, **kwargs)


gymenv = gym.make('Breakout-v0')
#gymenv = gym.make('Atlantis-v0')

q_model = get_or_create_model(Q_MODEL_PATH, gymenv.action_space.n, 'linear')
q_model2 = get_or_create_model(Q_MODEL_PATH + '.copy', gymenv.action_space.n, 'linear')
v_model = get_or_create_model(V_MODEL_PATH, 1, 'linear')

q_fn = ApproximateQFunction(KerasModel(q_model))
q_fn2 = ApproximateQFunction(KerasModel(q_model))
v_fn = ApproximateValueFunction(KerasModel(v_model))

policy = QEGreedyPolicy(q_fn)

#agent = DeepQAgent(q_fn, processor=AtariScreenProcessor(), policy=policy)
agent = DoubleDeepQAgent(q1=q_fn, q2=q_fn2, processor=AtariScreenProcessor(), policy=policy)

#agent = MonteCarloAgent(q=q_fn, v=v_fn, lr=0.5, processor=AtariScreenProcessor(), policy=policy)
#agent = TemporalDifferenceAgent(q=q_fn, v=v_fn, processor=AtariScreenProcessor(), policy=policy)

env = GymEnvironment(gymenv, agent)

step = 0
while True:
    env.step()
    env.render()
    step += 1
    if step % 10000 == 0:
        q_model.save(Q_MODEL_PATH)
        v_model.save(V_MODEL_PATH)
        print('Saving model')
