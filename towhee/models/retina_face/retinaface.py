# Copyright 2021 biubug6 . All rights reserved.
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

#adapted from https://github.com/biubug6/Pytorch_Retinaface
#from collections import OrderedDict
from typing import Tuple, Dict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision.models import _utils

from towhee.models.retina_face.ssh import SSH
from towhee.models.retina_face.mobilenet_v1 import MobileNetV1
from towhee.models.retina_face.retinaface_fpn import RetinaFaceFPN
from towhee.models.retina_face.heads import ClassHead, BboxHead, LandmarkHead
from towhee.models.retina_face.prior_box import PriorBox
from towhee.models.retina_face.utils import decode, decode_landm

class RetinaFace(nn.Module):
    """
    ReitinaFace

    RetinaFace: Single-stage Dense Face Localisation in the Wild.
    Described in https://arxiv.org/abs/1905.00641.

    Args:
        cfg (`Dict`):
            Network related settings.
        phase (`str`):
            train or test.
    """
    def __init__(self, cfg: Dict=None, phase: str='train'):
        super().__init__()
        self.phase = phase
        backbone = None
        self.cfg = cfg
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            #if cfg['pretrain']:
            #    checkpoint = torch.load('./pretrained_weights/mobilenetV1X0.25_pretrain.tar', map_location=torch.device('cpu'))
            #    new_state_dict = OrderedDict()
            #    for k, v in checkpoint['state_dict'].items():
            #        name = k[7:]  # remove module.
            #        new_state_dict[name] = v
            #    # load params
            #    backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            #backbone = models.resnet50(pretrained=cfg['pretrain'])
            backbone = models.resnet50(False)

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.max_size = cfg['max_size']
        self.target_size = cfg['target_size']
        self.fpn = RetinaFaceFPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.class_head = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.bbox_head = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.landmark_head = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for _ in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead

    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for _ in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num: int=3,inchannels: int=64,anchor_num :int=2):
        landmarkhead = nn.ModuleList()
        for _ in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs: torch.FloatTensor):

        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.bbox_head[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.class_head[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.landmark_head[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output

    def inference(self, img: np.array) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        im_shape = img.shape
        h, w, _ = im_shape
        im_height, im_width = h, w

        # pylint: disable=unsubscriptable-object
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        device = img.get_device()
        if device == -1:
            device = torch.device('cpu')
        mean = torch.FloatTensor(self.cfg['mean']).to(device)
        img = img - mean
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
        img = img.to(device)

        scale = scale.to(device)

        loc, conf, landms = self.forward(img)
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.to(device)
        scores = conf.squeeze(0).data[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])

        scale_replicate = torch.Tensor([img.shape[3], img.shape[2]] * 5)
        scale_replicate = scale_replicate.to(device)
        dets = torch.hstack((boxes, scores.unsqueeze(-1)))
        keep = torchvision.ops.nms(dets[:,:4], dets[:,4], self.cfg['nms_threshold'])
        dets = dets[keep, :]
        landms = landms[keep]
        confident_ids = torch.where(dets[:,4] > self.cfg['confidence_threshold'])
        return dets[confident_ids], landms[confident_ids]


