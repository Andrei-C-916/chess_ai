import torch.nn as nn

class ChessCNN(nn.Module):
    #model architecture: input -> conv2d -> relu -> conv2d -> relu -> conv2d -> relu -> flatten -> linear -> relu -> linear
    #input is an 8 x 8 matrix (representing a chess board) with 13 channels (12 for each unique piece and 1 for legal moves)
    #num_classes is the total number of unique moves in the dataset

    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(13,64,3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(64,128,3,stride=1,padding=1)

        self.fc1 = nn.Linear(8*8*128,512)
        self.fc2 = nn.Linear(512,num_classes)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x