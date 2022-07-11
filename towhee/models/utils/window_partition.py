# original code from https://github.com/SwinTransformer/Video-Swin-Transformer
# modified by Zilliz.


def window_partition(x, window_size: int):
    """
    Args:
       x: (b, h, w, c)
       window_size (int): window size
    Returns:
       windows: (num_windows*B, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows
