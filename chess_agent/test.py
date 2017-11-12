import chess
from chess_agent import model

values = []

m = model.Conv3DValueModel()

board = chess.Board()
values.append(m.predict_board_value(board))

board.push_san("e4")
values.append(m.predict_board_value(board))

board.push_san("e5")
values.append(m.predict_board_value(board))

board.push_san("Qh5")
values.append(m.predict_board_value(board))

board.push_san("Nc6")
values.append(m.predict_board_value(board))

board.push_san("Bc4")
values.append(m.predict_board_value(board))

board.push_san("Nf6")
values.append(m.predict_board_value(board))

board.push_san("Qxf7")
values.append(m.predict_board_value(board))

print(values)
