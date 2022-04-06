# original code from https://github.com/krasserm/perceiver-io
# modified by Zilliz


from towhee.models.perceiver.create_self_attention import create_self_attention
from towhee.models.perceiver.Sequential import Sequential


def create_self_attention_block(
    num_layers: int, num_channels: int, num_heads: int, dropout: float, activation_checkpoint: bool = False
):
    """
    Self attention block for Perceiver https://arxiv.org/pdf/2103.03206.pdf.
    Args:
        num_layers (`int`):
            Number of layers.
        num_channels (`int`):
            Number of channels.
        num_heads (`int`):
            Number of parallel attention heads.
        dropout (`nn.Module`):
            Dropout probability.
        activation_checkpoint (`bool`):
            Use activation checkpointing.
    """
    layers = [create_self_attention(num_channels, num_heads, dropout, activation_checkpoint)
              for _ in range(num_layers)]
    return Sequential(*layers)
