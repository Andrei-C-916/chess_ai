import torch
from game.chess_module import ChessModule
from models.alpha_resnet import ResNet
from .alphazero import AlphaZero

game = ChessModule()

num_input_channels = 13
num_hidden_channels = 64
num_resBlocks = 5

model = ResNet(
    game=game,
    num_input_channels=num_input_channels,
    num_hidden_channels=num_hidden_channels,
    num_resBlocks=num_resBlocks
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

args = {
    'num_iterations': 5,
    'num_selfPlay_iterations': 10,
    'num_epochs': 3,
    'batch_size': 32,
    'C': 1.0,
    'num_searches': 100,
    'dirichlet_alpha': 0.3,
    'dirichlet_eps': 0.25,
    'temperature_moves': 20,
    'temperature_early': 1.0,
    'temperature_late': 0.1
}

az = AlphaZero(
    model=model,
    optimizer=optimizer,
    game=game,
    args=args
)

az.learn()
