import os
import sys
import numpy as np
import chess
import chess.pgn
import chess_agent.model as model


class PGNLearner(object):
    BATCH_SIZE = 10000

    def __init__(self, m):
        self.model = m
        self.states = None
        self.rewards = None

    def get_states_from_pgn_game(self, game):
        print('Reading game: {0}'.format(game.headers['Event']))
        board = game.board()
        history = [board.copy()]
        reward = model.RESULT_MAP[game.headers['Result']]
        for move in game.main_line():
            board.push(move)
            history.append(board.copy())
        return self.model.data_from_history(history, reward)

    def read_states_from_pgn_file(self, filename):
        with open(filename) as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    return
                if 'Variant' in game.headers:
                    continue
                result = game.headers['Result']
                if result not in model.RESULT_MAP:
                    print('Unknown result: {0}. Skipping'.format(result))
                    continue
                if result == '1/2-1/2':
                    print('Draw. Skipping')
                    continue
                res = self.get_states_from_pgn_game(game)
                if res is not None:
                    states, rewards = res
                    self.append_states(states, rewards)

    def learn_pgn(self, path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                if name.endswith('.pgn'):
                    self.read_states_from_pgn_file(os.path.join(root, name))
        self.train_batch()

    def append_states(self, states, rewards):
        if self.states is None:
            self.states = states
        else:
            self.states = np.concatenate((self.states, states))

        if self.rewards is None:
            self.rewards = rewards
        else:
            self.rewards = np.concatenate((self.rewards, rewards))

        if len(self.states) > self.BATCH_SIZE:
            self.train_batch()

    def train_batch(self):
        self.model.train(self.states, self.rewards)
        self.model.save()
        print('Model saved')
        self.states = None
        self.rewards = None


if __name__ == '__main__':
    m = model.Conv3DValueModel()
    learner = PGNLearner(m)
    learner.learn_pgn(sys.argv[1])
