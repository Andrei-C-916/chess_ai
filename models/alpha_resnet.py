import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, game, num_input_channels, num_hidden_channels, num_resBlocks, kernel_size=3, padding=1):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(num_input_channels, num_hidden_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_hidden_channels),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden_channels, kernel_size=kernel_size, padding=padding) for _ in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden_channels, 32,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden_channels, num_input_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_input_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(num_input_channels * game.row_count * game.column_count, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden_channels, num_hidden_channels,
                               kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(num_hidden_channels)
        self.conv2 = nn.Conv2d(num_hidden_channels, num_hidden_channels,
                               kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(num_hidden_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = F.relu(x)
        return x
