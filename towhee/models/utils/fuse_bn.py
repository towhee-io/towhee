# Copyright 2022 Zilliz. All rights reserved.
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
#
# Inspired by https://github.com/DingXiaoH/RepMLP

from torch import nn


def fuse_bn(conv_or_fc: nn.Module, bn: nn.Module):
    std = (bn.running_var + bn.eps).sqrt()
    t = bn.weight / std
    t = t.reshape(-1, 1, 1, 1)

    if len(t) == conv_or_fc.weight.size(0):
        return conv_or_fc.weight * t, bn.bias - bn.running_mean * bn.weight / std
    else:
        repeat_times = conv_or_fc.weight.size(0) // len(t)
        repeated = t.repeat_interleave(repeat_times, 0)
        return conv_or_fc.weight * repeated, (bn.bias - bn.running_mean * bn.weight / std).repeat_interleave(
            repeat_times, 0)
