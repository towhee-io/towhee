# Implementation of model in the paper:
# Contrastive Learning with Large Memory Bank and Negative Embedding Subtraction for Accurate Copy Detection
# Paper link: https://arxiv.org/abs/2112.04323
# Inspired by the original code: https://github.com/lyakaap/ISC21-Descriptor-Track-1st.
#
# Copyright 2021 Zilliz. All rights reserved.
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

import torch.cuda
from torch import nn


class ISCNet(nn.Module):
    """
    CNN model of ISC.

    Args:
        backbone (`nn.Module`):
            Backbone module.
        fc_dim (`int=256`):
            Feature dimension of the fc layer.
        p (`float=3.0`):
            Power used in pooling for training.
        eval_p (`float=4.0`):
            Power used in pooling for evaluation.
    """
    def __init__(self, backbone, fc_dim=256, p=3.0, eval_p=4.0):
        super().__init__()

        self.backbone = backbone
        backbone_dim = [x.shape[0] for x in self.backbone.parameters()][-1]
        if hasattr(self.backbone, 'feature_info'):
            assert backbone_dim == self.backbone.feature_info.info[-1]['num_chs']
        self.fc = nn.Linear(backbone_dim, fc_dim, bias=False)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.p = p
        self.eval_p = eval_p

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.backbone(x)[-1]
        assert len(x.shape) == 4
        _, _, height, width = x.shape
        if torch.is_tensor(height):
            height = height.item()
            width = width.item()
        p = self.p if self.training else self.eval_p
        x = nn.functional.avg_pool2d(x.clamp(min=1e-6).pow(p), (height, width)).pow(1./p)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.bn(x)
        x = nn.functional.normalize(x)
        return x


def create_model(timm_backbone=None, pretrained=False, checkpoint_path=None, device=None, **kwargs):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if timm_backbone:
        import timm  # pylint: disable=C0415
        backbone = timm.create_model(timm_backbone, features_only=True, pretrained=False)
        kwargs.update(backbone=backbone)
    model = ISCNet(**kwargs).to(device)
    if pretrained:
        assert checkpoint_path, 'Checkpoint path is mandatory for pretrained model.'
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                state_dict[k[len('module.'):]] = state_dict[k]
                del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model
