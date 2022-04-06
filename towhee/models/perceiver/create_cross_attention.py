# original code from https://github.com/krasserm/perceiver-io
# modified by Zilliz


import os

try:
    import fairscale
except ModuleNotFoundError:
    os.system("pip install fairscale")
from fairscale.nn import checkpoint_wrapper

from towhee.models.perceiver.Residual import Residual
from towhee.models.perceiver.Sequential import Sequential
from towhee.models.perceiver.mlp import mlp
from towhee.models.perceiver.CrossAttention import CrossAttention


def cross_attention_layer(
    num_q_channels: int, num_kv_channels: int, num_heads: int, dropout: float, activation_checkpoint: bool = False
):
    """
    Cross attention block for Perceiver https://arxiv.org/pdf/2103.03206.pdf.

    Args:
        num_q_channels (`int`):
            Number of q channels.
        num_kv_channels (`int`):
            Number of k or v channels. k has the same channels as v.
        num_heads (`int`):
            Number of parallel attention heads.
        dropout (`nn.Module`):
            Dropout probability.
        activation_checkpoint (`bool`):
            Use activation checkpointing.

    Return (`nn.Module`):
        Configured cross attention layer.
    """
    layer = Sequential(
        Residual(CrossAttention(num_q_channels, num_kv_channels, num_heads, dropout), dropout),
        Residual(mlp(num_q_channels), dropout),
    )
    return layer if not activation_checkpoint else checkpoint_wrapper(layer)
