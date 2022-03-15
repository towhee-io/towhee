# original code from https://github.com/SwinTransformer/Video-Swin-Transformer
# modified by Zilliz.


def window_reverse(windows, window_size, b, d, h, w):
    """
    Args:
        windows (`torch.Tensor`):
            Tensor with size (B*num_windows, window_size, window_size, C)
        window_size (`tuple[int]`):
            3D window size.
        b (`int`):
            Batch size
        d (`int`):
            Window size in time dimension.
        h (`int`):
            Height of image
        w (`int`):
            Width of image
    Returns:
        x (`torch.Tensor`):
            Tensor with size (b, d, h, w, c)
    """
    x = windows.view(b, d // window_size[0], h // window_size[1], w // window_size[2],
                     window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)
    return x
