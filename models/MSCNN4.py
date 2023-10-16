import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------------------------------------------------------------------------------------------------------- #

"""
# Ori. Model: https://github.com/ShaneSpace/MyResearchWorksPublic/blob/main/reimplementation/model04_pengdandan_mmcnn.py
The functions (normalization_processing, wgn, add_noise, moving_average, gaussian_func, and gaussian_filtering) are for 
data preprocessing, noise addition, and filtering. They are not directly related to the architecture of the neural 
network model. When converting the model architecture from TensorFlow to PyTorch, the primary focus is on layers, 
connections, and forward pass computations.
"""

"""
1. Ensure that the data you feed into these functions is in the form of PyTorch tensors. If the data is still in numpy 
arrays, you can convert using torch.from_numpy(your_numpy_array). 
2. Functions return PyTorch tensors. If you  need  them as NumPy arrays, you can convert them using .numpy() method 
after detaching and moving to the CPU with .cpu( ).detach().numpy().
"""


def normalization_processing(data):
    data_mean = torch.mean(data)
    data_var = torch.var(data)
    normalized_data = (data - data_mean) / torch.sqrt(data_var)
    return normalized_data


def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = torch.sum(x ** 2) / x.numel()
    npower = xpower / snr
    noise = torch.randn_like(x) * torch.sqrt(npower)
    return noise


def add_noise(data, snr_num):
    rand_data = wgn(data, snr_num)
    noised_data = data + rand_data
    return noised_data


def moving_average(x, w=5):
    padding = w // 2
    weights = torch.ones(w) / w
    x_padded = torch.nn.functional.pad(x, (padding, padding))
    ma_data = torch.nn.functional.conv1d(x_padded.unsqueeze(0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0))
    return ma_data.squeeze()


def gaussian_func(x):
    pi_tensor = torch.tensor(math.pi)
    delta = torch.tensor(1.0)
    return 1 / (delta * torch.sqrt(2 * pi_tensor)) * torch.exp(-x ** 2 / (2 * delta ** 2))


def gaussian_filtering(x, w=5):
    w_j = torch.arange(w).float() - 2
    gaussian_coef = torch.stack([gaussian_func(i) for i in w_j])
    x_padded = torch.nn.functional.pad(x, (w // 2, w // 2))
    filtered_data = torch.nn.functional.conv1d(x_padded.unsqueeze(0).unsqueeze(0),
                                               gaussian_coef.unsqueeze(0).unsqueeze(0))
    return filtered_data.squeeze()


# -------------------------------------------------------------------------------------------------------------------- #

"""
Note: 
“K” is the first convolutional kernel length. 
“C” is the convolutional channel. 
“S” is the convolutional stride. 
“D” is the dropout rate.
"""


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_rate):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class MultiScaleModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout_rate):
        super(MultiScaleModule, self).__init__()
        kernel_sizes = [6, 12, 24, 48]  # Fixed kernel sizes based on the prior use of K and h.
        self.blocks = nn.ModuleList(
            [Conv1DBlock(in_channels, out_channels, k, stride, dropout_rate) for k in kernel_sizes])

    def forward(self, x):
        outputs = [block(x) for block in self.blocks]
        return torch.cat(outputs, dim=1)


class ConvBranch(nn.Module):
    def __init__(self, in_channel):
        super(ConvBranch, self).__init__()
        self.layers = nn.Sequential(
            Conv1DBlock(in_channel, 16, 6, 4, 0.5),
            Conv1DBlock(16, 32, 5, 2, 0.4),
            Conv1DBlock(32, 64, 4, 2, 0.3),
            Conv1DBlock(64, 128, 3, 2, 0.2),
            Conv1DBlock(128, 256, 2, 2, 0.1)
        )

    def forward(self, x):
        return self.layers(x)


class MultiScaleBranch(nn.Module):
    def __init__(self, in_channel):
        super(MultiScaleBranch, self).__init__()
        self.layers = nn.Sequential(
            MultiScaleModule(in_channel, 16, 4, 0.5),
            MultiScaleModule(64, 32, 2, 0.4),   # Output channels from the previous layer is 16*4=64
            MultiScaleModule(128, 64, 2, 0.3),  # 32*4=128
            MultiScaleModule(256, 128, 2, 0.2), # 64*4=256
            MultiScaleModule(512, 256, 2, 0.1)  # 128*4=512
        )

    def forward(self, x):
        return self.layers(x)


class MSCNN4(nn.Module):
    def __init__(self, in_channel=1, out_channel=15):
        super(MSCNN4, self).__init__()
        self.multiscale_branch = MultiScaleBranch(in_channel)
        self.conv_branch = ConvBranch(in_channel)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1280, out_channel)  # 8 comes from 4*256 from the MS and 256 from the conv branch.

    def forward(self, x):
        y1 = self.multiscale_branch(x)
        y2 = self.conv_branch(x)

        feature_map = torch.cat([y1, y2], dim=1)
        y = self.global_avg_pool(feature_map).squeeze(2)
        y = self.fc(y)
        output = F.softmax(y, dim=1)

        return feature_map, output
