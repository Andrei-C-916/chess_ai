import numpy as np
import chess

from chess_utils import board_to_encoding, generate_all_moves

def test_board_to_encoding_shape_and_dtype():
    board = chess.Board()
    tensor = board_to_encoding(board)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (13, 8, 8)

def test_board_to_encoding_legal_moves_plane():
    board = chess.Board()
    tensor = board_to_encoding(board)
    legal_to = {m.to_square for m in board.legal_moves}
    plane = tensor[12]
    ones = {(r * 8) + c for r in range(8) for c in range(8) if plane[r, c] == 1}
    assert ones == legal_to

def test_board_to_encoding_current_player_and_opponent_planes():
    board = chess.Board()
    tensor_white = board_to_encoding(board)
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        if piece.color == chess.WHITE:
            assert tensor_white[0:6, row, col].sum() == 1
            assert tensor_white[6:12, row, col].sum() == 0
        else:
            assert tensor_white[0:6, row, col].sum() == 0
            assert tensor_white[6:12, row, col].sum() == 1

    board.turn = chess.BLACK
    tensor_black = board_to_encoding(board)
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        if piece.color == chess.BLACK:
            assert tensor_black[0:6, row, col].sum() == 1
            assert tensor_black[6:12, row, col].sum() == 0
        else:
            assert tensor_black[0:6, row, col].sum() == 0
            assert tensor_black[6:12, row, col].sum() == 1

def test_generate_all_moves_size_and_uniqueness():
    moves = generate_all_moves()
    assert isinstance(moves, list)
    assert len(moves) == len(set(moves))
    assert len(moves) == 4608