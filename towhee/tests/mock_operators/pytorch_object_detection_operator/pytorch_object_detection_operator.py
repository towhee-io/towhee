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
import numpy
import torch
import torchvision

from typing import NamedTuple, List
from pathlib import Path
from towhee.operator import Operator

cache_path = Path(__file__).parent.parent.parent.resolve()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
lb_names = COCO_INSTANCE_CATEGORY_NAMES


class PyTorchObjectDetectionOperator(Operator):
    """
    PyTorch object detection model operator, detect the region of interests in one image and predict the object classes
    in these regions.
    """

    def __init__(self, model_name) -> None:
        super().__init__()
        model_func = getattr(torchvision.models.detection, model_name)
        self._model = model_func(pretrained=True)
        self._model.eval()

    def __call__(self, images: List[torch.Tensor], mask_score=0.5) -> NamedTuple('Outputs', [('boxes', numpy.ndarray),
                                                                           ('breed', str)]):
        Outputs = NamedTuple('Outputs', [('boxes', numpy.ndarray), ('breed', str)])
        outputs = self._model(images)
        pred = outputs[0]
        scores = pred['scores']
        mask = scores >= mask_score
        boxes = pred['boxes'][mask]
        labels = pred['labels'][mask]
        if labels is not None:
            breeds = [lb_names[i] for i in labels] if lb_names is not None else None

        return Outputs(boxes.detach().numpy(), breeds)

    def train(self):
        """
        For training model
        """
        pass
