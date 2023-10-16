import torch
import torch.nn as nn


class SelfAttention1D(nn.Module):
    def __init__(self, in_channels, attention_size):
        super(SelfAttention1D, self).__init__()
        self.query = nn.Conv1d(in_channels, attention_size, 1)
        self.key = nn.Conv1d(in_channels, attention_size, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        attn_weights = self.softmax(torch.bmm(q.transpose(1, 2), k))
        return x + torch.bmm(attn_weights, v.transpose(1, 2)).transpose(1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.skip = nn.Conv1d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.main(x) + self.skip(x))


class MSCNN7AM(nn.Module):
    def __init__(self, in_channel=1, out_channel=15, num_fc_neurons=None):
        super(MSCNN7AM, self).__init__()
        if num_fc_neurons is None:
            num_fc_neurons = [256, 64]
        self.scale2, self.scale3 = nn.AvgPool1d(2, 2), nn.AvgPool1d(3, 3)

        self.conv_blocks = nn.ModuleList([
            ResidualBlock(in_channel, 8),
            ResidualBlock(8, 16),
            ResidualBlock(16, 32),
            ResidualBlock(32, 64)
        ])

        self.attention = SelfAttention1D(64, 32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 3, num_fc_neurons[0]),
            nn.ReLU(inplace=True),
            nn.Linear(num_fc_neurons[0], num_fc_neurons[1]),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(num_fc_neurons[1], out_channel)

    def _forward_convs(self, x):
        for block in self.conv_blocks:
            x = block(x)
        return x

    def forward(self, x):
        x = x.view(-1, 1, 1024)
        x1, x2, x3 = map(self._forward_convs, [x, self.scale2(x), self.scale3(x)])
        x1, x2, x3 = map(self.attention, [x1, x2, x3])

        features = torch.cat([self.pool(x1), self.pool(x2), self.pool(x3)], dim=1).view(x.size(0), -1)
        conv_features = self.fc_layers(features)

        return conv_features, self.fc(conv_features)
