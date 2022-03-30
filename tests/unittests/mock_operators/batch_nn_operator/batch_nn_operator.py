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

import torch
from torch import nn

from towhee.operator import NNOperator, SharedType
from towhee.utils.log import engine_log
from towhee.serving import auto_batch

Outputs = NamedTuple("Outputs", [("res", int)])


@auto_batch(5, 0.01)
class BatchNnOperator(NNOperator):
    """
    A test NNOperator with no functionality.
    """

    def __init__(self, framework: str = 'pytorch'):
        super().__init__()
        self._framework = framework
        self.model = nn.Identity()

    def __call__(self, num: int, batch: int):
        t = torch.randn(num, num)
        if batch == 1:
            res = [self.model(t)]
        else:
            res = self.model([t] * batch)
        return Outputs(torch.stack(res).sum())

    @property
    def shared_type(self):
        return SharedType.Shareable
