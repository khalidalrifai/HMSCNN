# Import necessary modules from PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Depthwise Separable Convolution
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()

        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# Modify the StackConvBlock to use DepthwiseSeparableConv1d and reduce depth
class StackConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(StackConvBlock, self).__init__()

        self.conv1 = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, stride=stride,
                                              padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size, stride=stride,
                                              padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.xx = nn.Conv1d(in_channels, out_channels, 1, stride=stride, padding=0)

    def forward(self, x):
        C1 = F.relu(self.bn1(self.conv1(x)))
        C2 = self.bn2(self.conv2(C1))
        C2 += self.xx(x)
        y = F.relu(C2)
        return y


# Modify the MSCNN5 model architecture
class MSCNN5(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=15):
        super(MSCNN5, self).__init__()

        self.conv0 = DepthwiseSeparableConv1d(in_channel, 32, 7, padding=3)
        self.bn0 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.stacks = nn.ModuleDict({
            f'stack{size}': self._make_stack(size) for size in [3, 5]
        })
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 2, 512)
        self.fc2 = nn.Linear(512, out_channel)

    def _make_stack(self, kernel_size):
        return nn.Sequential(
            StackConvBlock(32, 64, kernel_size),
            StackConvBlock(64, 128, kernel_size),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x0 = F.relu(self.bn0(self.conv0(x)))
        x0 = self.pool(x0)
        feature_list = [stack(x0).squeeze(-1) for stack in self.stacks.values()]
        cat_feature_vec = torch.cat(feature_list, dim=1)
        cat_feature_vec = self.dropout(cat_feature_vec)
        x4 = F.relu(self.fc1(cat_feature_vec))
        features_map = self.fc2(x4)
        out = F.softmax(features_map, dim=1)
        return features_map, out
