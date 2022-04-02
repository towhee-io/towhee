# original code from https://github.com/krasserm/perceiver-io
# modified by Zilliz


from torch import nn

from MultiHeadAttention import MultiHeadAttention


class SelfAttention(nn.Module):
    """
    Self attention for Perceiver https://arxiv.org/pdf/2103.03206.pdf.
    Args:
        num_channels (`int`):
            Number of channels.
        num_heads (`int`):
            Number of parallel attention heads.
        dropout (`nn.Module`):
            Dropout probability.
    """
    def __init__(self, num_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_q_channels=num_channels, num_kv_channels=num_channels, num_heads=num_heads, dropout=dropout
        )

    def forward(self, x, pad_mask=None, attn_mask=None):
        """
        Forward function.
        Args:
            x (`Tensor`):
                Input tensor
            pad_mask (`Tensor`):
                Padding mask.
            attn_mask (`Tensor`):
                Attention mask.
        """
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)
