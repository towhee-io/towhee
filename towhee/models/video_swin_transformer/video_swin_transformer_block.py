# original code from https://github.com/SwinTransformer/Video-Swin-Transformer
# modified by Zilliz.

from torch import nn
import numpy as np
from einops import rearrange

from towhee.models.layers.swin_transformer_block3d import SwinTransformerBlock3D
from towhee.models.utils.get_window_size import get_window_size
from towhee.models.video_swin_transformer.compute_mask import compute_mask


class VideoSwinTransformerBlock(nn.Module):
    """
    A basic Swin Transformer Block for one stage.

    Args:
        dim (`int`):
            Number of feature channels
        depth (`int`):
            Depths of this stage.
        num_heads (`int`):
            Number of attention head.
        window_size (`tuple[int]`):
            Local window size. Default: (1,7,7).
        mlp_ratio (`float`):
            Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (`bool`):
            If True, add a learnable bias to query, key, value. Default: True
        qk_scale (`float`):
            Override default qk scale of head_dim ** -0.5 if set.
        drop (`float`):
            Dropout rate. Default: 0.0
        attn_drop (`float`):
            Attention dropout rate. Default: 0.0
        drop_path (`float`):
            Stochastic depth rate. Default: 0.0
        norm_layer (`nn.Module`):
            Normalization layer. Default: nn.LayerNorm
        downsample (`nn.Module`):
            Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """
        Forward function.

        Args:
            x (`torch.Tensor`):
                Input feature, tensor size (b, c, d, h, w).
        """
        # calculate attention mask for SW-MSA
        b, _, d, h, w = x.shape
        window_size, shift_size = \
            get_window_size((d, h, w), self.window_size, self.shift_size)  # pylint: disable=unbalanced-tuple-unpacking
        x = rearrange(x, 'b c d h w -> b d h w c')
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask(dp, hp, wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x
