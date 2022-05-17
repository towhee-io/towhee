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

import math
import torch
from torch import nn
import torchvision

from towhee.models.acar_net.utils import HR2ONL


class LinearHead(nn.Module):
    """
    Linear head of ACAR-Net
    """
    def __init__(self, width, roi_spatial=7, num_classes=60, dropout=0., bias=False):
        super().__init__()

        self.roi_spatial = roi_spatial
        self.roi_maxpool = nn.MaxPool2d(roi_spatial)

        self.fc = nn.Linear(width, num_classes, bias=bias)

        if dropout > 0:
            self.dp = nn.Dropout(dropout)
        else:
            self.dp = None

    # data: features, rois
    # returns: outputs
    def forward(self, data):
        if not isinstance(data['features'], list):
            features = [data['features']]
        else:
            features = data['features']

        roi_features = []
        for f in features:
            sp = f.shape
            h, w = sp[3:]
            feats = nn.AdaptiveAvgPool3d((1, h, w))(f).view(-1, sp[1], h, w)

            rois = data['rois'].clone()
            rois[:, 1] = rois[:, 1] * w
            rois[:, 2] = rois[:, 2] * h
            rois[:, 3] = rois[:, 3] * w
            rois[:, 4] = rois[:, 4] * h
            rois = rois.detach()
            roi_feats = torchvision.ops.roi_align(feats, rois, (self.roi_spatial, self.roi_spatial))
            roi_feats = self.roi_maxpool(roi_feats).view(-1, sp[1])

            roi_features.append(roi_feats)

        roi_features = torch.cat(roi_features, dim=1)
        if self.dp is not None:
            roi_features = self.dp(roi_features)
        outputs = self.fc(roi_features)

        return outputs


class ACARHead(nn.Module):
    """
    ACAR head of ACAR-Net
    """
    def __init__(self, width, roi_spatial=7, num_classes=60, dropout=0., bias=False,
                 reduce_dim=1024, hidden_dim=512, downsample='max2x2', depth=2,
                 kernel_size=3, mlp_1x1=False):
        super().__init__()

        self.roi_spatial = roi_spatial
        self.roi_maxpool = nn.MaxPool2d(roi_spatial)

        # actor-context feature encoder
        self.conv_reduce = nn.Conv2d(width, reduce_dim, 1, bias=False)

        self.conv1 = nn.Conv2d(reduce_dim * 2, hidden_dim, 1, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, bias=False)

        # down-sampling before HR2O
        assert downsample in ['none', 'max2x2']
        if downsample == 'none':
            self.downsample = nn.Identity()
        elif downsample == 'max2x2':
            self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # high-order relation reasoning operator (HR2ONL)
        layers = []
        for _ in range(depth):
            layers.append(HR2ONL(hidden_dim, kernel_size, mlp_1x1))
        self.hr2o = nn.Sequential(*layers)

        # classification
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(reduce_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim * 2, num_classes, bias=bias)

        if dropout > 0:
            self.dp = nn.Dropout(dropout)
        else:
            self.dp = None

    # data: features, rois, num_rois, roi_ids, sizes_before_padding
    # returns: outputs
    def forward(self, data):
        if not isinstance(data['features'], list):
            feats = [data['features']]
        else:
            feats = data['features']

        # temporal average pooling
        h, w = feats[0].shape[3:]
        # requires all features have the same spatial dimensions
        feats = [nn.AdaptiveAvgPool3d((1, h, w))(f).view(-1, f.shape[1], h, w) for f in feats]
        feats = torch.cat(feats, dim=1)

        feats = self.conv_reduce(feats)

        rois = data['rois']
        rois[:, 1] = rois[:, 1] * w
        rois[:, 2] = rois[:, 2] * h
        rois[:, 3] = rois[:, 3] * w
        rois[:, 4] = rois[:, 4] * h
        rois = rois.detach()
        roi_feats = torchvision.ops.roi_align(feats, rois, (self.roi_spatial, self.roi_spatial))
        roi_feats = self.roi_maxpool(roi_feats).view(data['num_rois'], -1)

        roi_ids = data['roi_ids']
        sizes_before_padding = data['sizes_before_padding']
        high_order_feats = []
        for idx in range(feats.shape[0]):  # iterate over mini-batch
            n_rois = roi_ids[idx+1] - roi_ids[idx]
            if n_rois == 0:
                continue

            eff_h, eff_w = math.ceil(h * sizes_before_padding[idx][1]), math.ceil(w * sizes_before_padding[idx][0])
            bg_feats = feats[idx][:, :eff_h, :eff_w]
            bg_feats = bg_feats.unsqueeze(0).repeat((n_rois, 1, 1, 1))
            actor_feats = roi_feats[roi_ids[idx]:roi_ids[idx+1]]
            tiled_actor_feats = actor_feats.unsqueeze(2).unsqueeze(2).expand_as(bg_feats)
            interact_feats = torch.cat([bg_feats, tiled_actor_feats], dim=1)

            interact_feats = self.conv1(interact_feats)
            interact_feats = nn.functional.relu(interact_feats)
            interact_feats = self.conv2(interact_feats)
            interact_feats = nn.functional.relu(interact_feats)

            interact_feats = self.downsample(interact_feats)

            interact_feats = self.hr2o(interact_feats)
            interact_feats = self.gap(interact_feats)
            high_order_feats.append(interact_feats)

        high_order_feats = torch.cat(high_order_feats, dim=0).view(data['num_rois'], -1)

        outputs = self.fc1(roi_feats)
        outputs = nn.functional.relu(outputs)
        outputs = torch.cat([outputs, high_order_feats], dim=1)

        if self.dp is not None:
            outputs = self.dp(outputs)
        outputs = self.fc2(outputs)

        return outputs

