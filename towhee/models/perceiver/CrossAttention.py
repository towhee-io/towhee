# original code from https://github.com/krasserm/perceiver-io
# modified by Zilliz


from torch import nn

from towhee.models.perceiver.MultiHeadAttention import MultiHeadAttention


class CrossAttention(nn.Module):
    """
    Cross attention for Perceiver https://arxiv.org/pdf/2103.03206.pdf.
    Args:
        num_q_channels (`int`):
            Number of q channels.
        num_kv_channels (`int`):
            Number of k or v channels. k has the same channels as v.
        num_heads (`int`):
            Number of parallel attention heads.
        dropout (`nn.Module`):
            Dropout probability.
    """
    def __init__(self, num_q_channels: int, num_kv_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attention = MultiHeadAttention(
            num_q_channels=num_q_channels, num_kv_channels=num_kv_channels, num_heads=num_heads, dropout=dropout
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, pad_mask=pad_mask, attn_mask=attn_mask)
