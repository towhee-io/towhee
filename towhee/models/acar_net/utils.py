# Code for paper:
# [Actor-Context-Actor Relation Network for Spatio-temporal Action Localization](https://arxiv.org/pdf/2006.07976.pdf)
#
# Original implementation by https://github.com/Siyu-C/ACAR-Net
#
# Modifications by Copyright 2021 Zilliz. All rights reserved.
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

from torch import nn


def get_bbox_after_aug(aug_info, bbox, aug_threshold=0.3):
    if aug_info is None:
        return bbox

    cbox = aug_info['crop_box']
    w = cbox[2] - cbox[0]
    h = cbox[3] - cbox[1]

    l = max(min(bbox[0], cbox[2]), cbox[0])
    r = max(min(bbox[2], cbox[2]), cbox[0])
    t = max(min(bbox[1], cbox[3]), cbox[1])
    b = max(min(bbox[3], cbox[3]), cbox[1])

    if (b - t) * (r - l) <= (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]) * aug_threshold:
        return None
    ret = [(l - cbox[0]) / w, (t - cbox[1]) / h, (r - cbox[0]) / w, (b - cbox[1]) / h]

    if aug_info['flip']:
        ret = [1. - ret[2], ret[1], 1. - ret[0], ret[3]]

    pad_ratio = aug_info['pad_ratio']
    ret = [ret[0] / pad_ratio[0], ret[1] / pad_ratio[1], ret[2] / pad_ratio[0], ret[3] / pad_ratio[1]]

    return ret


class HR2ONL(nn.Module):
    """
    HR2O_NL module for ACAR head
    """
    def __init__(self, hidden_dim=512, kernel_size=3, mlp_1x1=False):
        super().__init__()

        self.hidden_dim = hidden_dim

        padding = kernel_size // 2
        self.conv_q = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_k = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)

        self.conv = nn.Conv2d(
            hidden_dim, hidden_dim,
            1 if mlp_1x1 else kernel_size,
            padding=0 if mlp_1x1 else padding,
            bias=False
        )
        self.norm = nn.GroupNorm(1, hidden_dim, affine=True)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        query = self.conv_q(x).unsqueeze(1)
        key = self.conv_k(x).unsqueeze(0)
        att = (query * key).sum(2) / (self.hidden_dim ** 0.5)
        att = nn.Softmax(dim=1)(att)
        value = self.conv_v(x)
        virt_feats = (att.unsqueeze(2) * value).sum(1)

        virt_feats = self.norm(virt_feats)
        virt_feats = nn.functional.relu(virt_feats)
        virt_feats = self.conv(virt_feats)
        virt_feats = self.dp(virt_feats)

        x = x + virt_feats
        return x
