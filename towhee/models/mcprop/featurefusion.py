# Built on top of the original implementation at https://github.com/mesnico/Wiki-Image-Caption-Matching/blob/master/mcprop/model.py
#
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

import torch
from torch import nn
from torch.nn import functional as F


class FeatureFusion(nn.Module):
    """
    Depth aggregator
    Args:
        mode (str): aggregator
        img_feat_dim (int): image feature dimension
        txt_feat_dim (int): text feature dimension
        common_space_dim (int): common space dimension
    """
    def __init__(self, mode, img_feat_dim, txt_feat_dim, common_space_dim):
        super().__init__()
        self.mode = mode
        if mode == 'concat':
            pass #TODO
        elif mode == 'weighted':
            self.alphas = nn.Sequential(
                nn.Linear(img_feat_dim + txt_feat_dim, 512),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(512, 2))
            self.img_proj = nn.Linear(img_feat_dim, common_space_dim)
            self.txt_proj = nn.Linear(txt_feat_dim, common_space_dim)
            self.post_process = nn.Sequential(
                nn.Linear(common_space_dim, common_space_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(common_space_dim, common_space_dim)
            )

    def forward(self, img_feat, txt_feat):
        concat_feat = torch.cat([img_feat, txt_feat], dim=1)
        alphas = torch.sigmoid(self.alphas(concat_feat))    # B x 2
        img_feat_norm = F.normalize(self.img_proj(img_feat), p=2, dim=1)
        txt_feat_norm = F.normalize(self.txt_proj(txt_feat), p=2, dim=1)
        out_feat = img_feat_norm * alphas[:, 0].unsqueeze(1) + txt_feat_norm * alphas[:, 1].unsqueeze(1)
        out_feat = self.post_process(out_feat)
        return out_feat, alphas
