# original code from https://github.com/SwinTransformer/Video-Swin-Transformer
# modified by Zilliz.

import torch
from torch import nn
import torch.nn.functional as F


class PatchMerging3D(nn.Module):
    """
    3D Patch Merging Layer.
    Args:
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`):
            Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        Forward function of 3D Patch Merging Layer.
        Args:
            x (`tensor`):
                Input tensor with size (b, d, h, w, c)
        """
        _, _, h, w, _ = x.shape

        # padding
        pad_input = (h % 2 == 1) or (w % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # b d h/2 w/2 c
        x1 = x[:, :, 1::2, 0::2, :]  # b d h/2 w/2 c
        x2 = x[:, :, 0::2, 1::2, :]  # b d h/2 w/2 c
        x3 = x[:, :, 1::2, 1::2, :]  # b d h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b d h/2 w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x
