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

from towhee.models.acar_net.utils import get_bbox_after_aug


class BasicNeck(nn.Module):
    """
    Basic Neck of ACAR-Net
    """
    def __init__(self, aug_threshold=0., num_classes=60, multi_class=True):
        super().__init__()

        # threshold on preserved ratio of bboxes after cropping augmentation
        self.aug_threshold = aug_threshold

        self.num_classes = num_classes
        self.multi_class = multi_class

    # data: aug_info, labels, filenames, mid_times
    # returns: num_rois, rois, roi_ids, targets, sizes_before_padding, filenames, mid_times, bboxes, bbox_ids
    def forward(self, data):
        rois, roi_ids, targets, sizes_before_padding, filenames, mid_times = [], [0], [], [], [], []
        bboxes, bbox_ids = [], []  # used for multi-crop fusion

        cur_bbox_id = -1  # record current bbox no.
        for idx in range(len(data['aug_info'])):
            aug_info = data['aug_info'][idx]
            pad_ratio = aug_info['pad_ratio']
            sizes_before_padding.append([1. / pad_ratio[0], 1. / pad_ratio[1]])

            for label in data['labels'][idx]:
                cur_bbox_id += 1
                bbox_list = [label['bounding_box']]

                for b in bbox_list:
                    bbox = get_bbox_after_aug(aug_info, b, self.aug_threshold)
                    if bbox is None:
                        continue
                    rois.append([idx] + bbox)

                    filenames.append(data['filenames'][idx])
                    mid_times.append(data['mid_times'][idx])
                    bboxes.append(label['bounding_box'])
                    bbox_ids.append(cur_bbox_id)

                    if self.multi_class:
                        ret = torch.zeros(self.num_classes)
                        ret.put_(torch.LongTensor(label['label']),
                                 torch.ones(len(label['label'])))
                    else:
                        ret = torch.LongTensor(label['label'])
                    targets.append(ret)

            roi_ids.append(len(rois))

        num_rois = len(rois)
        if num_rois == 0:
            return {'num_rois': 0, 'rois': None, 'roi_ids': roi_ids, 'targets': None,
                    'sizes_before_padding': sizes_before_padding,
                    'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes, 'bbox_ids': bbox_ids}

        rois = torch.FloatTensor(rois).cpu()
        targets = torch.stack(targets, dim=0).cpu()
        return {'num_rois': num_rois, 'rois': rois, 'roi_ids': roi_ids, 'targets': targets,
                'sizes_before_padding': sizes_before_padding,
                'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes, 'bbox_ids': bbox_ids}
