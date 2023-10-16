import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCNN3(nn.Module):
    def __init__(self, in_channel=1, out_channel=15):
        super(MSCNN3, self).__init__()

        # Define multiple convolutional layers with increasing kernel sizes
        # For each convolutional layer, also define a corresponding BatchNormalization and MaxPooling layer
        self.conv1, self.bn1, self.pool1 = self._conv_bn_pool_layer(in_channel, 32, kernel_size=2, stride=2,
                                                                    pool_size=256)
        self.conv2, self.bn2, self.pool2 = self._conv_bn_pool_layer(in_channel, 32, kernel_size=4, stride=4,
                                                                    pool_size=128)
        self.conv3, self.bn3, self.pool3 = self._conv_bn_pool_layer(in_channel, 32, kernel_size=8, stride=8,
                                                                    pool_size=64)
        self.conv4, self.bn4, self.pool4 = self._conv_bn_pool_layer(in_channel, 32, kernel_size=16, stride=16,
                                                                    pool_size=32)
        self.conv5, self.bn5, self.pool5 = self._conv_bn_pool_layer(in_channel, 32, kernel_size=32, stride=32,
                                                                    pool_size=16)
        self.conv6, self.bn6, self.pool6 = self._conv_bn_pool_layer(in_channel, 32, kernel_size=64, stride=64,
                                                                    pool_size=8)
        self.conv7, self.bn7, self.pool7 = self._conv_bn_pool_layer(in_channel, 32, kernel_size=128, stride=128,
                                                                    pool_size=4)
        self.conv8, self.bn8, self.pool8 = self._conv_bn_pool_layer(in_channel, 32, kernel_size=256, stride=256,
                                                                    pool_size=2)

        # Fully connected layers to get final output
        self.fc1 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(0.5)  # 50% dropout to reduce overfitting
        self.fc2 = nn.Linear(128, out_channel)

    def _conv_bn_pool_layer(self, in_channel, out_channel, kernel_size, stride, pool_size):
        """Utility function to define a set of Convolution, BatchNormalization, and MaxPooling layers."""
        conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
        bn = nn.BatchNorm1d(out_channel)
        pool = nn.MaxPool1d(kernel_size=pool_size)
        return conv, bn, pool

    def forward(self, x):
        # Pass the input tensor through each set of layers (conv -> batch norm -> relu -> pool)
        x1 = self.pool1(self.bn1(F.relu(self.conv1(x))))
        x2 = self.pool2(self.bn2(F.relu(self.conv2(x))))
        x3 = self.pool3(self.bn3(F.relu(self.conv3(x))))
        x4 = self.pool4(self.bn4(F.relu(self.conv4(x))))
        x5 = self.pool5(self.bn5(F.relu(self.conv5(x))))
        x6 = self.pool6(self.bn6(F.relu(self.conv6(x))))
        x7 = self.pool7(self.bn7(F.relu(self.conv7(x))))
        x8 = self.pool8(self.bn8(F.relu(self.conv8(x))))

        # Concatenate the results from all convolutional layers
        xx = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=2)

        # Flatten the tensor and pass through fully connected layers
        xx = xx.view(xx.size(0), -1)
        features = F.relu(self.fc1(xx))
        features = self.dropout(features)  # Apply dropout
        output = F.softmax(self.fc2(features), dim=1)  # Use softmax for classification

        return features, output
