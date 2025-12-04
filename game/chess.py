from auxillary import generate_all_moves, board_to_tensor
from chess.pgn import Game
from chess import Board
import chess.pgn
import numpy as np

ALL_MOVES = generate_all_moves()
MOVE_TO_INT = {move: i for i, move in enumerate(ALL_MOVES)}
INT_TO_MOVE = {i: move for i, move in enumerate(ALL_MOVES)}

class Chess:
    def __init__(self):
        self.action_size = len(ALL_MOVES)

    def get_initial_state(self):
        return Board()
    
    def get_next_state(self, state, action):
        pass

    def get_state_tensor(self, state):
        return board_to_tensor(state)