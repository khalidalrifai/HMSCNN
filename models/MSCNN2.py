import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCNN2(nn.Module):
    def __init__(self, in_channel=1, out_channel=15):
        super(MSCNN2, self).__init__()

        # First convolutional block:
        # Convolution: Takes 'in_channel' input channels and produces 16 output channels.
        # Kernel size is 64 and stride is 8. No padding is used.
        # Followed by batch normalization, ReLU activation, and max pooling.
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, 64, stride=8, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Second convolutional block:
        # Convolution: Takes 16 input channels and produces 32 output channels.
        # Kernel size is 3, stride is 1. No padding is used.
        # Followed by batch normalization, ReLU activation, and max pooling.
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, 3, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Third convolutional block:
        # Similar to the second, but the output channels increase to 64.
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, 3, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Fourth convolutional block:
        # Takes 64 input channels and produces 64 output channels.
        # Kernel size, stride, and other configurations are the same as previous blocks.
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, 3, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Fifth convolutional block:
        # Configuration remains the same as the fourth block.
        self.conv5 = nn.Sequential(
            nn.Conv1d(64, 64, 3, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # To compute the output size after the convolutional blocks,
        # a dummy input is passed through the convolutional layers.
        # This helps in determining the input size for the first fully connected layer.
        dummy_x = torch.randn(1, in_channel, 1024)
        f5 = self.forward_conv(dummy_x)
        self.flattened_size = f5.view(-1).size(0)

        # Two fully connected layers:
        # The first one connects the output of the convolutional blocks to 100 neurons.
        # The second one connects those 100 neurons to 'out_channel' output neurons.
        self.fc1 = nn.Linear(self.flattened_size, 100)
        self.fc2 = nn.Linear(100, out_channel)

    def forward_conv(self, x):
        # Pass the input through all convolutional blocks.
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        f5 = self.conv5(f4)
        return f5

    def forward(self, x):
        # Pass the input through the convolutional blocks.
        features_map = self.forward_conv(x)

        # Flatten the output from the convolutional blocks.
        x = features_map.view(features_map.size(0), -1)

        # Pass the flattened tensor through the fully connected layers.
        x = F.relu(self.fc1(x))
        out = self.fc2(x)

        # Return the output of the last convolutional block and the final output.
        return features_map, out
