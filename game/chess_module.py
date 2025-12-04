from chess_utils import generate_all_moves, board_to_tensor
from chess import Board
import numpy as np

ALL_MOVES = generate_all_moves()
UCI_TO_INT = {uci: i for i, uci in enumerate(ALL_MOVES)}
INT_TO_UCI = {i: uci for i, uci in enumerate(ALL_MOVES)}

class ChessModule:
    def __init__(self):
        self.action_size = len(ALL_MOVES)

    def get_initial_state(self):
        return Board()
    
    def get_next_state(self, state: Board, move: int):
        state.push_uci(INT_TO_UCI[move])
        return state

    def get_state_tensor(self, state: Board):
        return board_to_tensor(state)