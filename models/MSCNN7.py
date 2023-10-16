import torch
import torch.nn as nn
import warnings


# Custom Activation Function
class Swish(nn.Module):
    """
    Implements the Swish activation function.
    Swish: f(x) = x * sigmoid(x)
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


# Convolutional Block
class ConvBlock(nn.Module):
    """
    Defines a convolutional block that consists of:
    Convolution -> Batch Normalization -> Swish Activation -> Dropout -> Max Pooling
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, strides):
        super(ConvBlock, self).__init__()

        # Convolution layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=strides)

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_channels, track_running_stats=False)

        # Swish activation function
        self.relu = Swish()

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Max pooling layer
        self.pool = nn.MaxPool1d(2, stride=2, return_indices=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x


# Multi-Scale Convolutional Neural Network
class MSCNN7(nn.Module):
    """
    Multi-Scale Convolutional Neural Network (MSCNN) with three scales.
    """

    def __init__(self, pretrained=False, in_channel=1, out_channel=15):
        super(MSCNN7, self).__init__()

        # Warning if trying to use a pretrained version
        if pretrained:
            warnings.warn("Pretrained model is not available")

        # Multi-Scale Convolutional Layers
        self.scale2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.scale3 = nn.AvgPool1d(kernel_size=3, stride=3)

        # Define ConvBlock layers for each scale
        self.layer1_blocks = nn.ModuleList(self._create_conv_blocks(in_channel))
        self.layer2_blocks = nn.ModuleList(self._create_conv_blocks(in_channel))
        self.layer3_blocks = nn.ModuleList(self._create_conv_blocks(in_channel))

        # Adaptive Average Pooling for each scale
        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.pool3 = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layers
        self.fc_layers = self._create_fc_layers()
        self.fc = nn.Linear(64, out_channel)

    def _create_conv_blocks(self, in_channel):
        """
        Helper function to create a list of ConvBlocks for a scale.
        """
        return [
            ConvBlock(in_channels=in_channel, out_channels=16, kernel_size=3, padding='same', strides=1),
            ConvBlock(in_channels=16, out_channels=32, kernel_size=3, padding='same', strides=1),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, padding='same', strides=1),
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, padding='same', strides=1)
        ]

    def _create_fc_layers(self):
        """
        Helper function to create the fully connected layers.
        """
        return nn.Sequential(
            nn.Linear(128 * 3, 256),
            Swish(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            Swish(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        # Reshape the input
        x = x.view(-1, 1, 1024)
        x2 = self.scale2(x)
        x3 = self.scale3(x)

        # Apply the convolutional blocks to each scale
        x1 = self._apply_conv_blocks(x, self.layer1_blocks)
        x2 = self._apply_conv_blocks(x2, self.layer2_blocks)
        x3 = self._apply_conv_blocks(x3, self.layer3_blocks)

        # Apply adaptive average pooling and concatenate
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)
        out = torch.cat([x1, x2, x3], dim=1)

        # Flatten and pass through the fully connected layers
        out = out.view(out.size(0), -1)
        conv_features = self.fc_layers(out)
        out = self.fc(conv_features)

        return conv_features, out

    def _apply_conv_blocks(self, x, conv_blocks):
        """
        Helper function to apply a list of ConvBlocks to an input.
        """
        for block in conv_blocks:
            x = block(x)
        return x
