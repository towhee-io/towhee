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

from typing import NamedTuple
from pathlib import Path

import torch
from torch import nn

from towhee.operator import NNOperator, SharedType
from towhee.utils.log import engine_log

from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from timm.models.factory import create_model

Outputs = NamedTuple("Outputs", [("res", int)])


class BatchNnOperator(NNOperator):
    """
    A test NNOperator with no functionality.
    """

    def __init__(self, model_name, num_classes=1000, framework: str = 'pytorch'):
        super().__init__(framework)
        self.model = create_model(model_name, pretrained=False, num_classes=num_classes)
        self.model.eval()
        self._model_handler = Path(__file__).absolute().parent / 'handler.py'

    def __call__(self, img: 'towhee._types.Image'):
        ret = self.predict(img)
        return Outputs(ret)

    @property
    def shared_type(self):
        return SharedType.Shareable


# if __name__ == '__main__':
#     op = BatchNnOperator('resnet50')
#     op.initialize('test', {'batch_size': 2})
#     from towhee._types.image import Image
#     import cv2
#     ndarray_img = cv2.imread('/Users/jiangjunjie/WorkSpace/images/1.png')
#     rgb_img = cv2.cvtColor(ndarray_img, cv2.COLOR_BGR2RGB)

#     img = Image(rgb_img, 'RGB')
#     out = op(img)
#     print(out)
