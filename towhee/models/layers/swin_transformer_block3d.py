# original code from https://github.com/SwinTransformer/Video-Swin-Transformer
# modified by Zilliz.

import torch
from torch import nn
from torch.utils import checkpoint
import torch.nn.functional as F

from towhee.models.layers.window_attention3d import WindowAttention3D
from towhee.models.utils.window_partition import window_partition
from towhee.models.utils.window_reverse import window_reverse
from towhee.models.layers.mlp import Mlp
from towhee.models.utils.get_window_size import get_window_size
from towhee.models.layers.droppath import DropPath


class SwinTransformerBlock3D(nn.Module):
    """
    3D Swin Transformer Block.

    Args:
        dim (`int`):
            Number of input channels.
        num_heads (`int`):
            Number of attention heads.
        window_size (`tuple[int]`):
            Window size.
        shift_size (`tuple[int]`):
            Shift size for SW-MSA.
        mlp_ratio (`float`):
            Ratio of mlp hidden dim to embedding dim.
        qkv_bias (`bool`):
            If True, add a learnable bias to query, key, value. Default: True
        qk_scale (`float`):
            Override default qk scale of head_dim ** -0.5 if set.
        drop (`float`):
            Dropout rate. Default: 0.0
        attn_drop (`float`):
            Attention dropout rate. Default: 0.0
        drop_path(`float`):
            Stochastic depth rate. Default: 0.0
        act_layer (`nn.Module`):
            Activation layer. Default: nn.GELU
        norm_layer (`nn.Module`):
            Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        b, d, h, w, c = x.shape
        window_size, shift_size = \
            get_window_size((d, h, w), self.window_size, self.shift_size)  # pylint: disable=unbalanced-tuple-unpacking

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # b*nW, Wd*Wh*Ww, c
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # b*nW, Wd*Wh*Ww, c
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, b, dp, hp, wp)  # b d' h' w' c
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :d, :h, :w, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x (`tensor`):
                Input feature, tensor size (B, D, H, W, C).
            mask_matrix (`tuple[int]`):
                Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x
