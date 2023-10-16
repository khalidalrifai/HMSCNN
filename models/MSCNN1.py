# Importing necessary modules from PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F


# Defining the modified model again
class Conv1D_BN_LeakyReLU_MaxPool_Skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv1D_BN_LeakyReLU_MaxPool_Skip, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.4)
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # Add a 1x1 convolution for the skip connection to adjust the number of channels
        self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        x_shortcut = self.skip_conv(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        # Adjust the size of x_shortcut using slicing (if needed) and then add
        x += x_shortcut[:, :, :x.size(2)]

        return x


class MSCNN1(nn.Module):
    def __init__(self, in_channel=1, out_channel=15):
        super(MSCNN1, self).__init__()

        self.conv1 = nn.Conv1d(in_channel, 2, kernel_size=100)
        self.conv2 = nn.Conv1d(in_channel, 2, kernel_size=200)
        self.conv3 = nn.Conv1d(in_channel, 2, kernel_size=300)
        self.conv4 = nn.Conv1d(in_channel, 2, kernel_size=400)

        self.conv_block1 = Conv1D_BN_LeakyReLU_MaxPool_Skip(8, 16, kernel_size=8, stride=2, padding=0)
        self.conv_block2 = Conv1D_BN_LeakyReLU_MaxPool_Skip(16, 32, kernel_size=32, stride=4, padding=0)
        self.conv_block3 = Conv1D_BN_LeakyReLU_MaxPool_Skip(32, 64, kernel_size=16, stride=2, padding=0)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, out_channel)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        max_length = max(x1.size(2), x2.size(2), x3.size(2), x4.size(2))

        x1 = F.pad(x1, (0, max_length - x1.size(2)))
        x2 = F.pad(x2, (0, max_length - x2.size(2)))
        x3 = F.pad(x3, (0, max_length - x3.size(2)))
        x4 = F.pad(x4, (0, max_length - x4.size(2)))

        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.view(x.size(0), -1)

        features = self.fc1(x)
        output = F.softmax(self.fc2(features), dim=1)

        return features, output
