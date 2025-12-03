import torch
import torch.nn as nn
from torch.optim import SGD
import funcs
from cnn_model import ChessCNN
from torch.utils.data import DataLoader, Dataset
import chess.pgn
import numpy as np
from dataset import ChessDataset
from tqdm import tqdm

pgn = open('./data/lichess_elite_2020-08.pgn')

print("processing games...")
games = []
i = 0
while True and i<=1000:
    game = chess.pgn.read_game(pgn)
    if game is None:
        break
    else:
        games.append(game)
    i += 1
print("games processed")

print("converting games to input...")
X, y = funcs.games_to_input(games)
y, moves_to_int = funcs.encode_moves(y)
print("games converted")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

print(f"number of inputs = {len(X)}")
num_classes = len(moves_to_int)
print(f"num_classes = {num_classes}")

dataset = ChessDataset(X,y)
model = ChessCNN(num_classes=num_classes)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

loss_fn = nn.CrossEntropyLoss()

optimizer = SGD(model.parameters(), lr=0.1)

for epoch in range(100):

    model.train()
   
    total_loss = 0

    for input, label in tqdm(dataloader):
        output = model(input)
        loss = loss_fn(output, label)
        loss.backward()
        total_loss += float(loss)
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"loss for epoch{epoch} = {total_loss}")

torch.save(model.state_dict(), "./test_checkpoint")
    



