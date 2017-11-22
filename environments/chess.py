import chess
from environments import Environment


class ChessEnv(Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.board = chess.Board()

    def get_available_actions(self):
        return [x for x in self.board.legal_moves]

    def reset(self):
        self.board = chess.Board()

    def do_action(self, action):
        self.board.push(action)

    def get_observations(self):
        return self.board