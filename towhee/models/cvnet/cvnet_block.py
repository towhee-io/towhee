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
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import interpolate as resize
from towhee.models.layers.conv4d import CenterPivotConv4d as Conv4d
from towhee.models.cvnet.cvnet_utils import Geometry
import math


class CVLearner(nn.Module):
    """
    CVLearner
    """
    def __init__(self, inch):
        super().__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, query_strides, key_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(key_strides)

            building_block_layers = []
            for idx, (outch, ksz, query_stride, key_stride) in enumerate(
                    zip(out_channels, kernel_sizes, query_strides, key_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (query_stride,) * 2 + (key_stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3, outch4 = 16, 32, 64, 128

        self.block1 = make_building_block(inch[1], [outch1], [5], [2], [2])
        self.block2 = make_building_block(outch1, [outch1, outch2], [3, 3], [1, 2], [1, 2])
        self.block3 = make_building_block(outch2, [outch2, outch2, outch3], [3, 3, 3], [1, 1, 2], [1, 1, 2])
        self.block4 = make_building_block(outch3, [outch3, outch3, outch4], [3, 3, 3], [1, 1, 1], [1, 1, 1])

        self.mlp = nn.Sequential(nn.Linear(outch4, outch4), nn.ReLU(), nn.Linear(outch4, 2))

    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def interpolate_query_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_ha, o_wa = spatial_size
        hypercorr = hypercorr.view(bsz, ha, wa, ch, o_ha, o_wa).permute(0, 3, 1, 2, 4, 5).contiguous()
        return hypercorr

    def forward(self, corr):
        # Encode correlation from each layer (Squeezing building blocks)
        out_block1 = self.block1(corr)
        out_block2 = self.block2(out_block1)
        out_block3 = self.block3(out_block2)
        out_block4 = self.block4(out_block3)

        # Predict logits with the encoded 4D-tensor
        bsz, ch, _, _, _, _ = out_block4.size()
        out_block4_pooled = out_block4.view(bsz, ch, -1).mean(-1)
        logits = self.mlp(out_block4_pooled).squeeze(-1).squeeze(-1)

        return logits


class Correlation:
    """
    Correlation
    """
    @classmethod
    def compute_crossscale_correlation(cls, src_feats, trg_feats, origin_resolution):
        """ Build 6-dimensional correlation tensor """
        eps = 1e-8

        bsz, ha, wa, hb, wb = origin_resolution

        # Build multiple 4-dimensional correlation tensor
        corr6d = []
        for src_feat in src_feats:
            ch = src_feat.size(1)
            sha, swa = src_feat.size(-2), src_feat.size(-1)
            src_feat = src_feat.view(bsz, ch, -1).transpose(1, 2)
            src_norm = src_feat.norm(p=2, dim=2, keepdim=True)

            for trg_feat in trg_feats:
                shb, swb = trg_feat.size(-2), trg_feat.size(-1)
                trg_feat = trg_feat.view(bsz, ch, -1)
                trg_norm = trg_feat.norm(p=2, dim=1, keepdim=True)

                corr = torch.bmm(src_feat, trg_feat)

                corr_norm = torch.bmm(src_norm, trg_norm) + eps
                corr = corr / corr_norm

                correlation = corr.view(bsz, sha, swa, shb, swb).contiguous()
                corr6d.append(correlation)

        # Resize the spatial sizes of the 4D tensors to the same size
        for idx, correlation in enumerate(corr6d):
            corr6d[idx] = Geometry.interpolate4d(correlation, [ha, wa, hb, wb])

        # Build 6-dimensional correlation tensor
        corr6d = torch.stack(corr6d).view(len(src_feats) * len(trg_feats), bsz, ha, wa, hb, wb).transpose(0, 1)

        return corr6d.clamp(min=0)

    @classmethod
    def build_crossscale_correlation(cls, query_feats, key_feats, scales, conv2ds):

        bsz, _, hq, wq = query_feats.size()
        bsz, _, hk, wk = key_feats.size()

        # Construct feature pairs with multiple scales
        query_feats_scalewise = []
        key_feats_scalewise = []
        for scale, conv in zip(scales, conv2ds):
            shq = round(hq * math.sqrt(scale))
            swq = round(wq * math.sqrt(scale))
            shk = round(hk * math.sqrt(scale))
            swk = round(wk * math.sqrt(scale))

            query_feats_out = conv(resize(query_feats, (shq, swq), mode='bilinear', align_corners=True))
            key_feats_out = conv(resize(key_feats, (shk, swk), mode='bilinear', align_corners=True))

            query_feats_scalewise.append(query_feats_out)
            key_feats_scalewise.append(key_feats_out)

        corrs = cls.compute_crossscale_correlation(query_feats_scalewise, key_feats_scalewise, (bsz, hq, wq, hk, wk))

        return corrs.contiguous()

