import os
import chess
import chess.pgn
import chess_agent.model as model


SELF_PLAY_GAMES = 10000
OUTPUT_PGN_PATH = 'selfplay.pgn'
MAX_MOVES = 100
SKIP_DRAWS = True
UPDATE_MODEL = False
RANDOMNESS = 0.9


def play_game(m):
    game = chess.pgn.Game()
    node = game

    b = chess.Board()
    history = [b.copy()]
    move_i = 0
    while not b.is_game_over():
        move, value = model.get_move(m, b, RANDOMNESS)
        node = node.add_variation(move, comment='V:{0:.2}'.format(value))
        b.push(move)
        history.append(b.copy())
        move_i += 1
        if move_i > MAX_MOVES:
            print('Too many moves. Skipping')
            return

    result = b.result()
    game.headers["Event"] = "Selfplay ({0}) [{1}]".format(move_i, m.DEFAULT_PATH)
    game.headers["Result"] = result
    game.headers["White"] = "RL White"
    game.headers["Black"] = "RL Black"
    reward = model.RESULT_MAP[result]

    if reward == 0:
        print('Draw. Skipping')
        #return

    if b.is_checkmate():
        print('Checkmate')

    if UPDATE_MODEL:
        states, rewards = m.data_from_history(history, reward)
        m.train(states, rewards)

    with open(OUTPUT_PGN_PATH, 'a+') as pgn_file:
        pgn_file.write(str(game))
        pgn_file.write('\n\n')
    print("Finished")


if __name__ == '__main__':
    m = model.Conv3DValueModel()
    for i in range(1, SELF_PLAY_GAMES):
        print('Playing game {0}/{1}'.format(i, SELF_PLAY_GAMES))
        play_game(m)
        if UPDATE_MODEL and i % 100 == 0:
            m.save()
            print("Model saved")
    m.save()
    print("Model saved")



