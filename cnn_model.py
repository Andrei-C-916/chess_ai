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
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)
    

class ResBlockSimple(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out += self.shortcut(x)
        out = self.relu(out)
        return self.relu(out)
    

class ChessResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.initial_conv = nn.Conv2d(13,64,kernel_size=3,padding=1)

        self.resblock1 = ResBlock(64,64)
        self.resblock2 = ResBlock(64,128)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8*8*128,512)
        self.fc2 = nn.Linear(512,num_classes)

    def forward(self,x):
        x = self.relu(self.initial_conv(x))

        x = self.resblock1(x)
        x = self.resblock2(x)

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    

class ChessResNetSimple(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.initial_conv = nn.Conv2d(13,64,kernel_size=3,padding=1)

        self.resblock1 = ResBlockSimple(64,128)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8*8*128,256)
        self.fc2 = nn.Linear(256,num_classes)

    def forward(self,x):
        x = self.relu(self.initial_conv(x))

        x = self.resblock1(x)

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    



    