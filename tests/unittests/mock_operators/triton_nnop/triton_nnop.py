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
from towhee import register
from towhee._types import Image
from towhee.operator import NNOperator

from .model import Model
from .preprocess import Preprocess
from .postprocess import Postprocess


@register(
    input_schema=[(Image, (-1, -1, 3))],
    output_schema=[(np.float32, (512, ))]
)
class TritonNnop(NNOperator):
    '''
    Mock nnoperator to test triton tools.
    '''
    def __init__(self, model_name: str):
        super().__init__()
        self.device = 'cpu'
        self.model = Model(model_name, self.device)
        self.preprocess = Preprocess()
        self.postprocess = Postprocess()

    def __call__(self, image: 'Image'):
        image_tensor = self.preprocess(image)
        features = self.model(image_tensor)
        return self.postprocess(features)

    def save_model(self, model_type, output_file, args):
        self.model(model_type, output_file, args)
