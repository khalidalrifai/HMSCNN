import warnings
from torch import nn


# =============================================================================
# 1D Convolutional Neural Network (CNN) Model for Vibrational Signal Processing
# =============================================================================
# The purpose of this model is to process vibrational signals that often
# originate from machines or equipment. Analyzing these signals provides
# insights into the health and operational status of the machine.

class CNN1D(nn.Module):

    def __init__(self, pretrained=False, in_channel=1, out_channel=15):
        """
        Initialize the CNN model.

        Parameters:
        - pretrained: Boolean, whether to use a pretrained model.
        - in_channel: Int, number of input channels (default is 1).
        - out_channel: Int, number of output channels (default is 15).
        """
        super(CNN1D, self).__init__()

        # Check if a pretrained model is requested
        if pretrained:
            warnings.warn("Pretrained Model Is Not Available")

        # Define Convolutional Layers
        self.layer1 = self._create_conv_layer(in_channel, 16, 3)
        self.layer2 = self._create_conv_layer(16, 32, 3, pooling=True)
        self.layer3 = self._create_conv_layer(32, 64, 3)
        self.layer4 = self._create_conv_layer(64, 128, 3, adaptive_pooling=True)

        # Define Fully Connected Layers
        self.layer5 = self._create_fc_layers()
        self.fc = nn.Linear(64, out_channel)

    def _create_conv_layer(self, in_channels, out_channels, kernel_size, pooling=False, adaptive_pooling=False):
        """
        Utility function to create a convolutional layer with optional pooling.
        """
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if pooling:
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        if adaptive_pooling:
            layers.append(nn.AdaptiveMaxPool1d(4))

        return nn.Sequential(*layers)

    def _create_fc_layers(self):
        """
        Utility function to create fully connected layers.
        """
        return nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Returns:
        - features_map: Features extracted from the processed signal.
        - out: Output of the network (classification or regression results).
        """
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        # Flatten the output for FC layers
        features_map = self.layer5(l4.view(l4.size(0), -1))

        out = self.fc(features_map)
        return features_map, out
