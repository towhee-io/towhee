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

import numpy as np
from towhee._types import Image


class Preprocess:

    def __call__(self, img: 'ndarray') -> 'ndarray':
        return np.random.rand(1, 3, 224, 224)

    def input_schema(self):
        return [(Image, (-1, -1, 3))]

    def output_schema(self):
        return [(np.float32, (1, 3, 224, 224))]
