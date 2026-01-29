from .chess_utils import generate_all_moves, board_to_encoding
from chess import Board
import numpy as np

ALL_MOVES = generate_all_moves()
UCI_TO_INT = {uci: i for i, uci in enumerate(ALL_MOVES)}
INT_TO_UCI = {i: uci for i, uci in enumerate(ALL_MOVES)}

class ChessModule:
    def __init__(self) -> int:
        self.action_size = len(ALL_MOVES)
        self.row_count = 8
        self.column_count = 8

    def get_initial_state(self) -> Board:
        return Board()
    
    def get_next_state(self, state: Board, move: int) -> Board:
        state.push_uci(INT_TO_UCI[move])
        return state

    def check_termination_and_get_value(self, state: Board) -> tuple[bool, int]:
        if state.is_game_over():
            result = state.result()
            if result == "1/2-1/2":
                return True, 0
            elif result == "1-0":
                return True, 1
            elif result == "0-1":
                return True, -1
        return False, 0
    
    def get_valid_moves(self, state: Board) -> np.ndarray:
        valid_moves = np.zeros(self.action_size, dtype=np.int8)
        for move in list(state.legal_moves):
            uci = move.uci()
            valid_moves[UCI_TO_INT[uci]] = 1
        return valid_moves

    def get_state_encoding(self, state: Board) -> np.ndarray:
        return board_to_encoding(state)