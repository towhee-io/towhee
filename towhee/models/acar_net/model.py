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

import torch
from torch import nn

from towhee.models.acar_net import backbone, neck, head


class AcarNet(nn.Module):
    """
    ACAR-Net built with backbone, neck, head

    Args:
        - configs (`dict`):
            A dictionary of parameters.
    """
    def __init__(self, configs: dict):
        super().__init__()
        self.configs = configs

        self.backbone = backbone(**configs['backbone'])
        self.neck = neck(**configs['neck'])
        self.head = head(**configs['head'])

    def forward(self, data):
        noaug_info = [{'crop_box': [0., 0., 1., 1.], 'flip': False, 'pad_ratio': [1., 1.]}] * len(data['labels'])
        i_n = {'aug_info': noaug_info, 'labels': data['labels'],
               'filenames': data['filenames'], 'mid_times': data['mid_times']}
        o = self.neck(i_n)

        output_list = [None] * len(o['filenames'])
        cnt_list = [0] * len(o['filenames'])

        for no in range(len(data['clips'])):
            i_b = data['clips'][no]
            o_b = self.backbone(i_b)

            i_n = {'aug_info': data['aug_info'][no], 'labels': data['labels'],
                   'filenames': data['filenames'], 'mid_times': data['mid_times']}
            o_n = self.neck(i_n)

            if o_n['num_rois'] == 0:
                continue
            ids = o_n['bbox_ids']

            i_h = {'features': o_b, 'rois': o_n['rois'],
                   'num_rois': o_n['num_rois'], 'roi_ids': o_n['roi_ids'],
                   'sizes_before_padding': o_n['sizes_before_padding']}
            o_h = self.head(i_h)

            outputs = o_h
            for idx in range(o_n['num_rois']):
                if cnt_list[ids[idx]] == 0:
                    output_list[ids[idx]] = outputs[idx]
                else:
                    output_list[ids[idx]] += outputs[idx]
                cnt_list[ids[idx]] += 1

        num_rois, filenames, mid_times, bboxes, targets, outputs = 0, [], [], [], [], []
        for idx in range(len(o['filenames'])):
            if cnt_list[idx] == 0:
                continue
            num_rois += 1
            filenames.append(o['filenames'][idx])
            mid_times.append(o['mid_times'][idx])
            bboxes.append(o['bboxes'][idx])
            targets.append(o['targets'][idx])
            outputs.append(output_list[idx] / float(cnt_list[idx]))

        if num_rois == 0:
            return {'outputs': None, 'targets': None, 'num_rois': 0,
                    'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes}

        final_outputs = torch.stack(outputs, dim=0)
        final_targets = torch.stack(targets, dim=0)
        return {'outputs': final_outputs, 'targets': final_targets, 'num_rois': num_rois,
                'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes}
