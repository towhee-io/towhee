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


import torch
import cv2
from PIL import Image
from typing import NamedTuple
from towhee.operator import Operator


class PytorchYoloDetectorOperator(Operator):
    """
    Stateful operator
    """

    def __init__(self, model_name) -> None:
        super().__init__()
        self._model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True)

    def __call__(self, img_path: str) -> NamedTuple("Outputs", [("objs_list", list)]):
        self.img_path = img_path
        # Get object detection results in yolov5 model
        results = self._model([img_path])
        self.bboxes = results.xyxy[0].tolist()
        # Get the detected objects list([PIL.Image])
        objs_list = self.get_obj_image()
        Outputs = NamedTuple("Outputs", [("objs_list", list)])
        return Outputs(objs_list)

    def get_obj_image(self):
        objs_list = []
        img = cv2.imread(self.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert image from openCV format to PIL.Image
        for bbox in self.bboxes:
            tmp_obj = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            pil_img = Image.fromarray(cv2.cvtColor(tmp_obj, cv2.COLOR_BGR2RGB))
            objs_list.append(pil_img)
        return objs_list
