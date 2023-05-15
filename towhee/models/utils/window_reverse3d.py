# original code from https://github.com/SwinTransformer/Video-Swin-Transformer
# Modifications by Copyright 2022 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
