import numpy as np
from chess.pgn import Game
from chess import Board
from typing import List
import chess

#returns a tensor representation of a chess board. Tensor is of shape (13,8,8)
#requires: board is of type Board
def board_to_tensor(board: Board):
    tensor = np.zeros((13,8,8))
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row,col = divmod(square,8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        tensor[piece_type + piece_color, row, col] = 1
    
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square,8)
        tensor[12,row_to,col_to] = 1

    return tensor

#returns an np.array of board tensors and an np.array of labels, where the board tensors are (13,8,8) and the labels are uci formatted strings.
#label y_i is the move that was played in position X_i
#requires: games is of type List[Game]
def games_to_input(games: List[Game]):
    X = []
    y = []
    for game in games:
        board = game.board() 
        for move in game.mainline_moves():
            X.append(board_to_tensor(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)

#returns an np.array of moves encoded as ints, a dict mapping moves to ints, and a dict mapping ints to moves.
#requires: moves is a list of uci formatted strings
def encode_moves(moves):
    unique_moves = list(set(moves))
    move_to_int = {move: int for int, move in enumerate(unique_moves)}
    int_to_move = {int: move for int, move in enumerate(unique_moves)}
    moves = [move_to_int[move] for move in moves]
    return np.array(moves, dtype=np.float32), move_to_int, int_to_move

def fen_and_moves_to_input(fen,moves):
    X = []
    y = [] 
    board = Board(fen)
    uci_list = moves.split()
    board.push(chess.Move.from_uci(uci_list[0]))
    uci_list = uci_list[1:]
    for uci in uci_list:
        tensor = board_to_tensor(board)
        label = uci
        X.append(tensor)
        y.append(label)
        board.push(chess.Move.from_uci(uci))
    return np.array(X, dtype=np.float32), np.array(y)



