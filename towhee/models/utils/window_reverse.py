# original code from https://github.com/SwinTransformer/Video-Swin-Transformer
# modified by Zilliz.


def window_reverse(windows, window_size: int, h: int, w: int):
    """
    Args:
       windows: (num_windows*b, window_size, window_size, c)
       window_size (int): Window size
       h (int): Height of image
       w (int): Width of image
    Returns:
       x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

