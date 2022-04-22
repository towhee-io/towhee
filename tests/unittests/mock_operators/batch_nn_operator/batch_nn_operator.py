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

from towhee.operator import NNOperator, SharedType

from timm.models.factory import create_model

Outputs = NamedTuple('Outputs', [('res', int)])


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
