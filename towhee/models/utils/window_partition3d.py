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

from functools import reduce
from operator import mul


def window_partition(x, window_size):
    """
    Args:
        x (`torch.Tensor`):
            Tensor with size of (b, d, h, w, c). b is batch size. d is time dimension size. h and w is frame size.
            c is channel size.
        window_size (`tuple[int]`):
            3d window size
    Returns:
        windows (`torch.Tensor`):
            Window partitioned tensor with size (B*num_windows, window_size*window_size, C)
    """
    b, d, h, w, c = x.shape
    x = x.view(b, d // window_size[0], window_size[0],
               h // window_size[1], window_size[1], w // window_size[2], window_size[2], c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), c)
    return windows
