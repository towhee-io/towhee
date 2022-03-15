# original code from https://github.com/SwinTransformer/Video-Swin-Transformer
# modified by Zilliz.

import torch
from functools import lru_cache

from towhee.models.utils.window_partition import window_partition


# cache each stage results
@lru_cache()
def compute_mask(d, h, w, window_size, shift_size, device):
    """
    Compute attention mask.

    Args:
        d (`int`):
            lenth in time space.
        h (`int`):
            height of image.
        w (`int`):
            width of image.
        window_size (`tuple[int]`):
            window size of W-MSA.
        shift_size (`tuple[int]`):
            shift size of window.
        device (`int`):
            device id.

    Returns:
        attn_mask (`torch.Tensor`):
            Computed attention mask.
    """
    img_mask = torch.zeros((1, d, h, w, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for di in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for hi in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for wi in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, di, hi, wi, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask
