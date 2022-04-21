# original code from https://github.com/krasserm/perceiver-io
# modified by Zilliz


from torch import nn


class MultiHeadAttention(nn.Module):
    """
    Multi head attention for Perceiver https://arxiv.org/pdf/2103.03206.pdf.
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
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        """
        Forward function.
        Args:
            x_q (`Tensor`):
                Query embeddings.
            x_kv (`Tensor`):
                Key embeddings. Key equals value.
            pad_mask (`int`):
                Padding mask.
            attn_mask (`nn.Module`):
                Attention mask.
        """
        return self.attention(x_q, x_kv, x_kv, key_padding_mask=pad_mask, attn_mask=attn_mask)[0]
