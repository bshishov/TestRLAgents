import random
from collections import deque

import gym
import numpy as np
import keras
from keras.layers import Dense, Activation, Conv2D, Dropout, Flatten, MaxPooling2D, Conv3D, MaxPooling3D
from keras.models import Sequential
from keras.optimizers import Adam
import keras.backend as K

LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95
EPSILON = 0.5
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.1
REPLAY_MEMORY_SIZE = 2000
BATCH_SIZE = 128
EPOCHS = 1

replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

env = gym.make('CartPole-v1')
#env = gym.make('Hopper-v1')
#env = gym.make('MountainCar-v0')
#env = gym.make('Acrobot-v1')
#env = gym.make('Pendulum-v0')

state = env.reset()


def dqn_loss(y_true, y_pred):
    return 0.5 * K.pow(y_true - y_pred, 2)


dqn = Sequential()
dqn.add(Dense(100, activation='relu', input_shape=env.observation_space.shape,))
dqn.add(Dense(100, activation='relu'))
dqn.add(Dense(100, activation='relu'))
dqn.add(Dense(env.action_space.n, activation='linear'))
dqn.compile(optimizer=Adam(lr=LEARNING_RATE), loss=dqn_loss)


step = 0
episode = 0
epsilon = EPSILON


def as_batch(input):
    return np.array([input, ])


while True:
    step += 1
    if episode > 0:
        env.render()

    # e-greedy exploration
    if np.random.rand() < epsilon or state is None:
        action = env.action_space.sample()
    else:
        action = np.argmax(dqn.predict(np.array([state, ]))[0])

    epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

    # Perfom an enviromental step
    new_state, reward, done, _ = env.step(action)

    replay_memory.append([state, action, reward, new_state, int(done)])

    if len(replay_memory) > BATCH_SIZE:
        for epoch in range(EPOCHS):
            batch = random.sample(replay_memory, BATCH_SIZE)
            batch = np.array(batch)

            states = np.array([x for x in batch[:, 0]])
            actions = np.array([x for x in batch[:, 1]])
            rewards = np.array([x for x in batch[:, 2]])
            new_states = np.array([x for x in batch[:, 3]])

            target_rewards = rewards + DISCOUNT_FACTOR * np.max(dqn.predict(new_states), axis=1)
            target_rewards[batch[:, 4] == 1] = rewards[batch[:, 4] == 1]
            current_q = dqn.predict(states)
            target_q = current_q.copy()
            for i in range(len(current_q)):
                target_q[i][actions[i]] = target_rewards[i]
            
            dqn.train_on_batch(states, target_q)

    state = new_state

    if done:
        print('Episode {0}'.format(episode))
        episode += 1
        state = env.reset()
