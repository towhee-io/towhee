# Copyright 2021  Facebook. All rights reserved.
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
# This code is modified by Zilliz.


import math


def round_width(width, multiplier, min_width=8, divisor=8, ceil=False):
    """
    Round width of filters based on width multiplier
    Args:
        width ('int'):
            The channel dimensions of the input.
        multiplier ('float'):
            The multiplication factor.
        min_width ('int'):
            The minimum width after multiplication.
        divisor ('int'):
            The new width should be dividable by divisor.
        ceil ('bool'):
            If True, use ceiling as the rounding method.
    """
    if not multiplier:
        return width

    width *= multiplier
    min_width = min_width or divisor
    if ceil:
        width_out = max(min_width, int(math.ceil(width / divisor)) * divisor)
    else:
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)
