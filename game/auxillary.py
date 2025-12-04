import numpy as np
from chess.pgn import Game
from chess import Board
from typing import List
import chess

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

def generate_all_moves():
    moves = set()
    board = chess.Board()

    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            for promo in [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                try:
                    move = chess.Move(from_sq, to_sq, promotion=promo)
                    if move in board.pseudo_legal:
                        moves.add(move.uci())
                except:
                    pass

    return sorted(list(moves))

