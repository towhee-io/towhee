# Copyright 2021 Ross Wightman . All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This code is modified by Zilliz.
from typing import Union

from torch import nn

from towhee.models.layers.conv2d_same import Conv2dSame
from towhee.models.layers.padding_functions import get_padding_value

def create_conv2d_pad(in_chs: int, out_chs: int, kernel_size: int, **kwargs) -> Union[nn.Conv2d, Conv2dSame]:
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        if 'num_experts' in kwargs:
            kwargs.pop('num_experts', '')
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)

