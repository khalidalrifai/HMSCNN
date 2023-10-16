import torch
import torch.nn as nn
import torch.nn.functional as F


# MultiHeadAttention module
class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, seq_len, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.seq_len = seq_len

        # To ensure the input dimension is divisible by the number of heads
        assert self.head_dim * num_heads == in_dim, "Input dimension must be divisible by the number of heads."

        self.query = nn.Linear(self.seq_len, self.seq_len)
        self.key = nn.Linear(self.seq_len, self.seq_len)
        self.value = nn.Linear(in_dim, self.head_dim * num_heads)
        self.fc_out = nn.Linear(self.head_dim * num_heads, in_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # Linear transformation for Q and K
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Reshape x for V
        x_reshaped = x.permute(0, 2, 1)

        # Linear transformation for V
        V = self.value(x_reshaped).view(batch_size, self.num_heads, -1, self.head_dim)

        # Scaled dot-product attention
        scaled_attention_logits = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_dim ** 0.5
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, V).permute(0, 2, 1, 3).contiguous()

        # Concatenate heads and pass through final linear transformation
        concatenated_output = output.view(batch_size, -1, self.num_heads * self.head_dim)
        final_output = self.fc_out(concatenated_output)

        return final_output


# MSCNN3AM with Multi-head Attention
class MSCNN3AM(nn.Module):
    def __init__(self, in_channel=1, out_channel=15, num_heads=8):
        super(MSCNN3AM, self).__init__()

        self.attns = nn.ModuleList([MultiHeadAttention(32, 2, num_heads) for _ in range(8)])

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channel, 32, kernel_size=2 ** i, stride=2 ** i) for i in range(1, 9)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(32) for _ in range(8)])
        self.pools = nn.ModuleList([
            nn.MaxPool1d(kernel_size=2 ** (9 - i)) for i in range(1, 9)
        ])
        self.fc1 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, out_channel)

    def forward(self, x):
        features_list = []
        for conv, bn, pool, attn in zip(self.convs, self.bns, self.pools, self.attns):
            out = F.relu(conv(x))
            out = bn(out)
            out = pool(out)
            out = attn(out)
            features_list.append(out)

        concatenated_features = torch.cat(features_list, dim=2)
        flattened = concatenated_features.view(concatenated_features.size(0), -1)
        fc_out = F.relu(self.fc1(flattened))
        fc_out = self.dropout(fc_out)
        output = F.softmax(self.fc2(fc_out), dim=1)

        return fc_out, output
