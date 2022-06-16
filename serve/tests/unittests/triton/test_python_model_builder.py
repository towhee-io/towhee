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

import serve.triton.type_gen as tygen
from serve.triton.python_model_builder import _from_tensor_to_obj, _from_obj_to_tensor
from towhee._types import Image


class TestPythonModelBuilder(unittest.TestCase):
    """
    Unit test for build_python_model.
    """

    def test_from_tensor_to_obj(self):
        type_info = tygen.ImageType.type_info(Image, [512, 512, 3], False)
        init_code = tygen.ImageType.init_code([512, 512, 3])

        lines = _from_tensor_to_obj(type_info, init_code, 'img', ['data_tensor', 'mode_tensor'])
        expected_results = ['img = towhee._types.Image(data_tensor.as_numpy(), str(mode_tensor.as_numpy()).decode(\'utf-8\'))']

        self.assertListEqual(expected_results, lines)

        type_info = tygen.ImageType.type_info(Image, [512, 512, 3], True)
        lines = _from_tensor_to_obj(type_info, init_code, 'img', ['data_tensor', 'mode_tensor'])
        expected_results = [
            'img = [towhee._types.Image(arg0.as_numpy(), str(arg1.as_numpy()).decode(\'utf-8\')) for arg0, arg1 in zip(data_tensor, mode_tensor)]']

        self.assertListEqual(expected_results, lines)

    def test_from_obj_to_tensor(self):
        type_info = tygen.ImageType.type_info(Image, [512, 512, 3], False)

        lines = _from_obj_to_tensor(type_info, 'img', ['data_tensor', 'mode_tensor'], ['OUTPUT0', 'OUTPUT1'])
        expected_results = [
            'data_tensor = pb_utils.Tensor(\'OUTPUT0\', numpy.array(img, numpy.int8))',
            'mode_tensor = pb_utils.Tensor(\'OUTPUT1\', numpy.array(img.mode, numpy.object_))'
        ]

        self.assertListEqual(expected_results, lines)

        type_info = tygen.ImageType.type_info(Image, [512, 512, 3], True)

        lines = _from_obj_to_tensor(type_info, 'img', ['data_tensor', 'mode_tensor'], ['OUTPUT0', 'OUTPUT1'])

        self.assertListEqual(expected_results, lines)


if __name__ == '__main__':
    unittest.main()
