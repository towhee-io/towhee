# original code from https://github.com/SwinTransformer/Video-Swin-Transformer
# modified by Zilliz.


def get_window_size(x_size, window_size, shift_size=None):
    """
    Args:
        x_size (`tuple[int]`):
            Tensor with size (B*num_windows, window_size, window_size, C)
        window_size (`tuple[int]`):
            3D window size.
        shift_size (`tuple[int]`):
            Shift size. ref: https://arxiv.org/pdf/2106.13230.pdf
    Returns (`tuple[int]`):
        shifted window size.
    """
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)
