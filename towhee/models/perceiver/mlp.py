# original code from https://github.com/krasserm/perceiver-io
# modified by Zilliz
from torch.nn import Sequential
from torch import nn


def mlp(num_channels: int):
    return Sequential(
        nn.LayerNorm(num_channels),
        nn.Linear(num_channels, num_channels),
        nn.GELU(),
        nn.Linear(num_channels, num_channels),
    )
