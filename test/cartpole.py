import gym

import numpy as np

from keras.layers import Dense, Activation, Conv2D, Dropout, Flatten, MaxPooling2D, Conv3D, MaxPooling3D
from keras.models import Sequential
from keras.optimizers import Adam
import keras.backend as K

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from environments.gym import GymEnvironment
from agents.dqn import DeepQAgent, DoubleDeepQAgent
from agents.mc_td import TemporalDifferenceAgent, MonteCarloAgent, Sarsa
from agents.policy import QEGreedyPolicy, QERandomPolicy
from agents.model import KerasModel, SkLearnModel
from agents.fn import ApproximateValueFunction, ApproximateQFunction

LEARNING_RATE = 0.001


def dqn_loss(y_true, y_pred):
    return 0.5 * K.pow(y_true - y_pred, 2)


gymenv = gym.make('CartPole-v1')
#gymenv = gym.make('MountainCar-v0')


def create_dqn(input_shape, outputs):
    dqn = Sequential()
    dqn.add(Dense(100, activation='relu', input_shape=input_shape,))
    dqn.add(Dense(100, activation='relu'))
    dqn.add(Dense(100, activation='relu'))
    dqn.add(Dense(outputs, activation='linear'))
    dqn.compile(optimizer=Adam(lr=LEARNING_RATE), loss=dqn_loss)
    return dqn


dqn = create_dqn(gymenv.observation_space.shape, gymenv.action_space.n)


q_deep = KerasModel(dqn)

q_linear = SkLearnModel(MLPRegressor(alpha=LEARNING_RATE))
v_linear = SkLearnModel(MLPRegressor(alpha=LEARNING_RATE))

q_linear.fit(np.zeros(gymenv.observation_space.shape), np.zeros(gymenv.action_space.n))
v_linear.fit(np.zeros(gymenv.observation_space.shape), 1)

q_fn = ApproximateQFunction(q_deep)
#q_fn = ApproximateQFunction(q_linear)
v_fn = ApproximateValueFunction(v_linear)
policy = QEGreedyPolicy(q_fn, epsilon=0.5, epsilon_min=0.1)

# DQN
#agent = DeepQAgent(q_deep)
#agent = DeepQAgent(q_linear)
#agent = DeepQAgent(q_fn, policy=policy)

# DDQN
dqn2 = create_dqn(gymenv.observation_space.shape, gymenv.action_space.n)
agent =DoubleDeepQAgent(q1=q_fn, q2=ApproximateQFunction(KerasModel(dqn2)), policy=policy)

#agent = Sarsa(q=q_fn, lr=0.9, discount=0.95, policy=policy)
#agent = TemporalDifferenceAgent(q=q_fn, v=v_fn, lr=0.5, policy=policy)
#agent = MonteCarloAgent(q=q_fn, v=v_fn, lr=0.5, policy=policy)

env = GymEnvironment(gymenv, agent)

while True:
    env.step()
    env.render()
    if env.is_done():
        env.reset()
