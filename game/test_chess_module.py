import chess_module
from chess_module import ChessModule
import chess

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
    tensor = chess_module.board_to_tensor(next_state)
    original_row, original_col = divmod(chess.A2, 8)
    new_row, new_col = divmod(chess.A3, 8)
    assert tensor[6][new_row][new_col] == 1
    assert tensor[:, original_row, original_col].sum() == 0
    pawn_plane_indices = [6]  
    opponent_pawn_planes_sum = sum(tensor[i][new_row][new_col] for i in pawn_plane_indices)
    assert opponent_pawn_planes_sum == 1
