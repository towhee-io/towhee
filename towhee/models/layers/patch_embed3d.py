# original code from https://github.com/SwinTransformer/Video-Swin-Transformer
# modified by Zilliz.

from torch import nn
import torch.nn.functional as F


class PatchEmbed3D(nn.Module):
    """
    3D patch embedding for video.

    Args:
        patch_size (`int`):
            Patch token size.
        c (`int`):
            Number of input video channels..
        embed_dim (`int`):
            Number of linear projection output channels.
        norm_layer (`nn.Module`):
            Normalization layer.
    """
    def __init__(self, patch_size=(2, 4, 4), c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.c = c
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(c, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # do padding
        _, _, d, h, w = x.size()
        if w % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
        if h % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
        if d % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))

        x = self.proj(x)  # b c d h w
        if self.norm is not None:
            d, h, w = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, d, h, w)

        return x
