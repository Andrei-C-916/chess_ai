import chess_module
from chess_module import ChessModule
import chess
import numpy as np

def test_action_size():
    game = ChessModule()
    assert game.action_size == 4608

def test_move_mappings():
    assert chess_module.INT_TO_UCI[chess_module.UCI_TO_INT["a1b2"]] == "a1b2"
    assert chess_module.UCI_TO_INT[chess_module.INT_TO_UCI[67]] == 67

def test_get_next_state():
    move = chess_module.UCI_TO_INT["a2a3"]
    game = ChessModule()
    initial_state = game.get_initial_state()
    next_state = game.get_next_state(initial_state, move)
    tensor = chess_module.board_to_encoding(next_state)
    original_row, original_col = divmod(chess.A2, 8)
    new_row, new_col = divmod(chess.A3, 8)
    assert tensor[6][new_row][new_col] == 1
    assert tensor[:, original_row, original_col].sum() == 0
    pawn_plane_indices = [6]  
    opponent_pawn_planes_sum = sum(tensor[i][new_row][new_col] for i in pawn_plane_indices)
    assert opponent_pawn_planes_sum == 1

def test_check_termination_and_get_value_draw():
    game = ChessModule()
    stalemate_board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    is_terminal, value = game.check_termination_and_get_value(stalemate_board)
    assert is_terminal is True
    assert value == 0

def test_check_termination_and_get_value_white_win():
    game = ChessModule()
    white_win_board = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    is_terminal, value = game.check_termination_and_get_value(white_win_board)
    assert is_terminal is True
    assert value == -1

def test_check_termination_and_get_value_black_win():
    game = ChessModule()
    black_win_board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    is_terminal, value = game.check_termination_and_get_value(black_win_board)
    assert is_terminal is True
    assert value == -1

def test_check_termination_and_get_value_ongoing():
    game = ChessModule()
    ongoing_board = game.get_initial_state()
    is_terminal, value = game.check_termination_and_get_value(ongoing_board)
    assert is_terminal is False
    assert value == 0

def test_get_valid_moves():
    game = chess_module.ChessModule()
    board = chess.Board()
    valid_mask = game.get_valid_moves(board)
    assert isinstance(valid_mask, np.ndarray)
    assert valid_mask.shape == (game.action_size,)
    assert valid_mask.dtype == np.int8
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        uci = move.uci()
        idx = chess_module.UCI_TO_INT[uci]
        assert valid_mask[idx] == 1

    legal_indices = {chess_module.UCI_TO_INT[m.uci()] for m in legal_moves}
    all_indices = set(range(game.action_size))
    illegal_indices = list(all_indices - legal_indices)
    sample_illegal = illegal_indices[:20]
    for idx in sample_illegal:
        assert valid_mask[idx] == 0

