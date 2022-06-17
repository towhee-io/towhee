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

import unittest
import numpy as np

from towhee._types.image import Image
from serve.triton.triton_config_builder import TritonModelConfigBuilder


class TestTritonModelConfigBuilder(unittest.TestCase):
    """
    Unit test for TritonModelConfigBuilder.
    """

    # pylint: disable=protected-access

    def test_get_triton_schema(self):
        input_annotations = [(Image, (512, 512, 3)), (int, ()), (str, (-1,))]
        output_annotations = [(np.float32, (1, 3, 224, 224))]

        input_schema = TritonModelConfigBuilder.get_input_schema(input_annotations)
        output_schema = TritonModelConfigBuilder.get_output_schema(output_annotations)
        expected_input_schema = {
            'INPUT0': ('TYPE_INT8', [512, 512, 3]),
            'INPUT1': ('TYPE_STRING', []),
            'INPUT2': ('TYPE_INT64', []),
            'INPUT3': ('TYPE_STRING', [-1])
        }
        expected_output_schema = {'OUTPUT0': ('TYPE_FP32', [1, 3, 224, 224])}
        self.assertDictEqual(input_schema, expected_input_schema)
        self.assertDictEqual(output_schema, expected_output_schema)
