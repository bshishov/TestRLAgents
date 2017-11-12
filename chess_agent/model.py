import os
import numpy as np
import chess

import keras
from keras.layers import Dense, Activation, Conv2D, Dropout, Flatten, MaxPooling2D, Conv3D, MaxPooling3D
from keras.models import Sequential
from chess_agent.processor import BoardProcessor, SixPlane2DProcessor, TwelvePlane3DProcessor


RANDOMNESS = 0.1
RESULT_MAP = {
    '1-0': +1,
    '0-1': -1,
    '1/2-1/2': 0
}


def get_move(m, b, randomness=RANDOMNESS):
    moves = [x for x in b.legal_moves]
    boards = []
    for move in moves:
        bb = b.copy()
        bb.push(move)
        boards.append(bb)

    bias = np.random.rand(len(moves)) * randomness - randomness * 0.5
    values = m.predict_boards_values(boards) + bias
    imax = np.argmax(values)
    return moves[imax], values[imax]


class ValueModel(object):
    DEFAULT_PATH = 'model.h5'
    MODEL_TRAIN_BATCH_SIZE = 128
    MODEL_TRAIN_EPOCHS = 5
    REWARD_DISCOUNT_FACTOR = 0.7
    LEARNING_RATE = 0.2

    model = None
    processor = None  # type: BoardProcessor

    def __init__(self, processor=None, path=DEFAULT_PATH):
        if processor is not None:
            self.processor = processor
        if os.path.exists(path):
            self.load(path)
        else:
            self.create()

    def load(self, path=DEFAULT_PATH):
        self.model = keras.models.load_model(path)

    def create(self):
        raise NotImplementedError

    def save(self, path=DEFAULT_PATH):
        if self.model is not None:
            self.model.save(path)
            self.model.save_weights(path + '.weights')

    def train(self, states, rewards):
        self.model.fit(states, rewards, epochs=self.MODEL_TRAIN_EPOCHS, batch_size=self.MODEL_TRAIN_BATCH_SIZE)

    def data_from_history(self, boards, outcome):
        white_states = []
        black_states = []
        for b in boards:
            if b.turn == chess.WHITE:
                white_states.append(self.processor.get_state(b, perspective=chess.WHITE))
            else:
                black_states.append(self.processor.get_state(b, perspective=chess.BLACK))
        wl, bl = len(white_states), len(black_states)
        white_rewards = np.zeros(shape=(wl,), dtype=np.float)
        black_rewards = np.zeros(shape=(bl,), dtype=np.float)

        # G(t) = R(t) + REWARD_DISCOUNT_FACTOR * R(t+1) + REWARD_DISCOUNT_FACTOR ^ 2 * R(t+2)...
        # We have reward only at t=T
        # so G(t) = R(T) * REWARD_DISCOUNT_FACTOR ^ (T - t)
        for t in range(wl):
            white_rewards[t] = outcome * np.power(self.REWARD_DISCOUNT_FACTOR, wl - t - 1)
        for t in range(bl):
            black_rewards[t] = -outcome * np.power(self.REWARD_DISCOUNT_FACTOR, bl - t - 1)

        white_values = self.predict_states_values(white_states)
        black_values = self.predict_states_values(black_states)

        # V(s) <- V(s) + lr * (Gt - V(s))
        white_values += self.LEARNING_RATE * (white_rewards - white_values)
        black_values += self.LEARNING_RATE * (black_rewards - black_values)

        return np.concatenate((white_states, black_states), axis=0), np.concatenate((white_values, black_values),
                                                                                    axis=0)

    def predict_board_value(self, board):
        return self.predict_boards_values([board])[0]

    def predict_boards_values(self, boards):
        states = [self.processor.get_state(b, perspective=b.turn) for b in boards]
        return self.predict_states_values(states)

    def predict_states_values(self, states):
        rewards = self.model.predict(np.array(states)).flatten()
        return rewards


class Conv2DValueModel(ValueModel):
    DEFAULT_PATH = 'conv2d_6plane.h5'
    processor = SixPlane2DProcessor()

    def create(self):
        initializer = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)

        model = Sequential()
        model.add(Conv2D(32, (5, 5),
                         padding='same',
                         input_shape=(8, 8, 6),
                         kernel_initializer=initializer,
                         name='conv_1'))
        model.add(Activation('relu', name='act_1'))
        model.add(Conv2D(32, (3, 3), name='conv_2', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(32, (2, 2), name='conv_3', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(256, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='tanh'))

        model.compile(optimizer='adam', loss='mse')
        self.model = model


class DenseValueModel(ValueModel):
    DEFAULT_PATH = 'dense_6plane.h5'
    processor = SixPlane2DProcessor()

    def create(self):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(256, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='tanh'))

        model.compile(optimizer='adam', loss='mse')
        self.model = model


class Conv3DValueModel(ValueModel):
    DEFAULT_PATH = 'conv3d_12plane.h5'
    processor = TwelvePlane3DProcessor()

    def create(self):
        planes = self.processor.output_shape[2]

        model = Sequential()
        model.add(Conv3D(32, (3, 3, planes),
                         padding='same',
                         input_shape=self.processor.output_shape,
                         name='conv3d_1'))
        #model.add(Activation('relu'))
        model.add(Conv3D(32, (2, 2, planes),
                         padding='same',
                         activation='relu',
                         name='conv3d_2'))
        # pool maximum across all planes
        model.add(MaxPooling3D(pool_size=(2, 2, planes), padding='valid'))
        #model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(512, activation='tanh'))
        #model.add(Dropout(0.2))
        model.add(Dense(1, activation='tanh'))

        model.compile(optimizer='adam', loss='mse')
        self.model = model



