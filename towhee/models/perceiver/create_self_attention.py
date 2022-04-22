# original code from https://github.com/krasserm/perceiver-io
# modified by Zilliz


import os

try:
    # pylint: disable=unused-import
    import fairscale
except ImportError:
    os.system("pip install fairscale")
from fairscale.nn import checkpoint_wrapper

from towhee.models.perceiver.sequential import Sequential
from towhee.models.perceiver.residual import Residual
from towhee.models.perceiver.self_attention import SelfAttention
from towhee.models.perceiver.mlp import mlp


def create_self_attention(num_channels: int, num_heads: int, dropout: float, activation_checkpoint: bool = False):
    """
    Self attention block for Perceiver https://arxiv.org/pdf/2103.03206.pdf.

    Args:
        num_channels (`int`):
            Number of q channels.
        num_heads (`int`):
            Number of parallel attention heads.
        dropout (`nn.Module`):
            Dropout probability.
        activation_checkpoint (`bool`):
            Use activation checkpointing.

    Return (`nn.Module`):
        Configured self attention layer.
    """
    layer = Sequential(
        Residual(SelfAttention(num_channels, num_heads, dropout), dropout), Residual(mlp(num_channels), dropout)
    )
    return layer if not activation_checkpoint else checkpoint_wrapper(layer)
