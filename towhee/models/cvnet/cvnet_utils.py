# Original pytorch implementation by:
# 'Correlation Verification for Image Retrieval'
#       - https://arxiv.org/abs/2204.01458
# Original code by / Copyright 2022, Seongwon Lee.
# Modifications & additions by / Copyright 2022 Zilliz. All rights reserved.
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
import torch.nn.functional as F


def get_configs(model_name='CVNet_R101'):
    args = {
        'CVNet_R101':
            {'resnet_depth': 101},
        'CVNet_R50':
            {'resnet_depth': 50},
    }
    return args[model_name]


def extract_feat_res_pycls(img, backbone, feat_ids, bottleneck_ids, lids):
    r""" Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone.stem(img)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__('s%d' % lid).__getattr__('b%d' % (bid+1)).f.forward(feat)

        if bid == 0:
            res = backbone.__getattr__('s%d' % lid).__getattr__('b%d' % (bid+1)).proj.forward(res)
            res = backbone.__getattr__('s%d' % lid).__getattr__('b%d' % (bid+1)).bn.forward(res)
        feat += res

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = backbone.__getattr__('s%d' % lid).__getattr__('b%d' % (bid+1)).relu.forward(feat)

    return feats


class Geometry(object):
    """
    Geometry
    """

    @staticmethod
    def interpolate4d(tensor4d, size):
        bsz, h1, w1, h2, w2 = tensor4d.size()
        ha, wa, hb, wb = size
        tensor4d = tensor4d.view(bsz, h1, w1, -1).permute(0, 3, 1, 2)
        tensor4d = F.interpolate(tensor4d, (ha, wa), mode='bilinear', align_corners=True)
        tensor4d = tensor4d.view(bsz, h2, w2, -1).permute(0, 3, 1, 2)
        tensor4d = F.interpolate(tensor4d, (hb, wb), mode='bilinear', align_corners=True)
        tensor4d = tensor4d.view(bsz, ha, wa, hb, wb)

        return tensor4d
