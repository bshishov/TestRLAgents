import chess
import numpy as np


def iter_pieces(board):
    for position in range(64):
        x, y = position % 8, position // 8
        piece = board.piece_at(position)
        if piece is None:
            continue
        yield piece, x, y


class BoardProcessor(object):
    output_shape = ()

    def get_state(self, board, perspective=chess.WHITE):
        raise NotImplementedError


class OnePlane2DProcessor(BoardProcessor):
    output_shape = (8, 8, 1)

    def get_state(self, board, perspective=chess.WHITE):
        # State as an image with a channel for each piece type (total 6)
        state = np.zeros(shape=self.output_shape, dtype=np.int8)
        for piece, x, y in iter_pieces(board):
            # +1 in cell for "my" pieces, -1 for opponents, considering the perspective
            col = +1 if piece.color == perspective else -1
            if perspective == chess.WHITE:
                state[x, y] = col * piece.piece_type
            else:
                # Flip the board vertically for black
                state[x, 7 - y] = col * piece.piece_type
        return state


class SixPlane2DProcessor(BoardProcessor):
    output_shape = (8, 8, 6)

    def get_state(self, board, perspective=chess.WHITE):
        # State as an image with a channel for each piece type (total 6)
        state = np.zeros(shape=self.output_shape, dtype=np.int8)
        for piece, x, y in iter_pieces(board):
            # +1 in cell for "my" pieces, -1 for opponents, considering the perspective
            col = +1 if piece.color == perspective else -1
            if perspective == chess.WHITE:
                state[x, y, piece.piece_type - 1] = col
            else:
                # Flip the board vertically for black
                state[x, 7 - y, piece.piece_type - 1] = col
        return state


class TwelvePlane3DProcessor(BoardProcessor):
    '''
    Board represented as 12 8x8 planes
    each plane for each type and piece color (6 pieces and 2 colors)
    1 - there is a piece
    0 - empty square
    '''
    output_shape = (8, 8, 12, 1)

    def get_state(self, board, perspective=chess.WHITE):
        state = np.zeros(shape=self.output_shape, dtype=np.int8)
        for piece, x, y in iter_pieces(board):
            if perspective == chess.BLACK:
                y = 7 - y
            if piece.color == perspective:
                state[x, y, piece.piece_type - 1] = 1
            else:
                state[x, y, 6 + piece.piece_type - 1] = 1
        return state
