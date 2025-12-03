from torch.utils.data import Dataset
import torch
import chess
from funcs import board_to_tensor
import os

#custom dataset
#X are the board tensors and y are the labels
class ChessDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return len(self.X)
    
class ChessPuzzleDataset(Dataset):
    def __init__(self, fen_list, moves_list, move_to_int):
        self.positions = []
        for fen, moves in zip(fen_list, moves_list):
            board = chess.Board(fen)
            uci_list = moves.split()
            board.push(chess.Move.from_uci(uci_list[0]))
            uci_list = uci_list[1:]
            for uci in uci_list:
                self.positions.append((board.fen(), uci))
                board.push(chess.Move.from_uci(uci))
        self.move_to_int = move_to_int

    def __getitem__(self, idx):
        fen, move = self.positions[idx]
        board = chess.Board(fen)
        board.push(chess.Move.from_uci(move))
        tensor = board_to_tensor(board)
        label = self.move_to_int[move]
        return torch.tensor(tensor, dtype=torch.float32), label
    
    def __len__(self):
        return len(self.positions)
    

class ChessPGNDataset(Dataset):
    def __init__(self, pgn_folder, move_to_int):
        self.positions=[]
        count = 0
        for filename in os.listdir(pgn_folder):
            if filename.endswith(".pgn"):
                with open(os.path.join(pgn_folder, filename)) as pgn_file:
                    while True:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break
                        board = game.board()
                        for move in game.mainline_moves():
                            fen = board.fen()
                            self.positions.append((fen, move.uci()))
                            board.push(move)
        self.move_to_int = move_to_int

    def __getitem__(self, idx):
        fen, move = self.positions[idx]
        board = chess.Board(fen)
        board.push(chess.Move.from_uci(move))
        tensor = board_to_tensor(board)
        label = self.move_to_int[move]
        return torch.tensor(tensor, dtype=torch.float32), label

    def __len__(self):
        return len(self.positions)
