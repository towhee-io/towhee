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

from towhee import register
import numpy as np
import logging

logger = logging.getLogger()


class MockModel:
    def __call__(self):
        return np.random.rand((1, 512))


@register(
    input_schema=[(np.float32, (-1, 3, 224, 224))],
    output_schema=[(np.float32, (-1, 512))]
)
class Model:
    """
    Mock model
    """
    def __init__(self, model_name: str, device: str = None):
        self._model_name = model_name
        self._device = device
        self._model = MockModel()

    @property
    def model(self):
        return self._model

    def __call__(self, image):
        return self.model(image)

    @property
    def optimizes(self):
        return []
