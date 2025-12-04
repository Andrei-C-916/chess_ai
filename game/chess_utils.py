import numpy as np
from chess import Board, Move, SQUARES, QUEEN, ROOK, BISHOP, KNIGHT

FILES = "abcdefgh"
RANKS = "12345678"

# returns a (13, 8, 8) tensor representation of [board]. Channels 1-6 are current player's pieces. 
# channels 7-12 are opposing player's pieces. Channel 13 is current player's legal moves
def board_to_encoding(board: Board):
    tensor = np.zeros((13, 8, 8))
    current_color = board.turn

    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        piece_idx = piece.piece_type - 1
        if piece.color == current_color:
            offset = 0
        else:
            offset = 6
        tensor[offset + piece_idx, row, col] = 1

    for move in board.legal_moves:
        row_to, col_to = divmod(move.to_square, 8)
        tensor[12, row_to, col_to] = 1

    return tensor

# returns a set of all possible UCI strings. Not all UCI strings are possible moves, 
# however these will get masked out when we consider legal moves.
def generate_all_moves():
    moves = set()
    for f1 in FILES:
        for r1 in RANKS:
            for f2 in FILES:
                for r2 in RANKS:
                    uci = f1 + r1 + f2 + r2
                    moves.add(uci)
                    if r1 == "7" and r2 == "8":
                        for p in "qrbn":
                            moves.add(uci + p)
                    if r1 == "2" and r2 == "1":
                        for p in "qrbn":
                            moves.add(uci + p)
    return sorted(moves)

