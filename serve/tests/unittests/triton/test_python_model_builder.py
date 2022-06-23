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

import serve.triton.type_gen as tygen
from serve.triton.python_model_builder import PyModelBuilder, gen_model_from_pickled_callable, gen_model_from_op
from towhee._types import Image
from tempfile import TemporaryDirectory

from . import EXPECTED_FILE_PATH


class TestPythonModelBuilder(unittest.TestCase):
    """
    Unit test for build_python_model.
    """

    # pylint: disable=protected-access

    def test_from_tensor_to_obj(self):
        type_info = tygen.ImageType.type_info(Image, [512, 512, 3], False)
        init_code = tygen.ImageType.init_code([512, 512, 3])

        lines = PyModelBuilder._from_tensor_to_obj(type_info, init_code, 'img', ['data_tensor', 'mode_tensor'])
        expected_results = ['img = towhee._types.Image(data_tensor.as_numpy(), mode_tensor.as_numpy()[0].decode(\'utf-8\'))']

        self.assertListEqual(expected_results, lines)

        type_info = tygen.ImageType.type_info(Image, [512, 512, 3], True)
        lines = PyModelBuilder._from_tensor_to_obj(type_info, init_code, 'img', ['data_tensor', 'mode_tensor'])
        expected_results = [
            'img = [towhee._types.Image(arg0.as_numpy(), arg1.as_numpy()[0].decode(\'utf-8\')) for arg0, arg1 in zip(data_tensor, mode_tensor)]']

        self.assertListEqual(expected_results, lines)

    def test_from_obj_to_tensor(self):
        type_info = tygen.ImageType.type_info(Image, [512, 512, 3], False)

        lines = PyModelBuilder._from_obj_to_tensor(type_info, 'img', ['data_tensor', 'mode_tensor'], ['OUTPUT0', 'OUTPUT1'])
        expected_results = [
            'data_tensor = pb_utils.Tensor(\'OUTPUT0\', numpy.array(img, numpy.int8))',
            'mode_tensor = pb_utils.Tensor(\'OUTPUT1\', numpy.array([img.mode], numpy.object_))'
        ]

        self.assertListEqual(expected_results, lines)

        type_info = tygen.ImageType.type_info(Image, [512, 512, 3], True)

        lines = PyModelBuilder._from_obj_to_tensor(type_info, 'img', ['data_tensor', 'mode_tensor'], ['OUTPUT0', 'OUTPUT1'])

        self.assertListEqual(expected_results, lines)

    def test_pickle_callable_pymodel_builder(self):
        with TemporaryDirectory(dir='./') as gen_file_path:
            pyfile_name = 'clip_preprocess_model.py'
            save_path = gen_file_path + '/' + pyfile_name
            gen_model_from_pickled_callable(
                save_path= save_path,
                module_name='towhee.operator.triton_nnop',
                python_file_path='triton_nnop/__init__.py',
                pickle_file_name='preprocess.pickle',
                input_annotations=[(Image, (512, 512, 3))],
                output_annotations=[(np.float32, (1, 3, 224, 224))]
            )

            with open(save_path, 'rt', encoding='utf-8') as gen_f, \
                 open(EXPECTED_FILE_PATH + pyfile_name, 'rt', encoding='utf-8') as expected_f:
                self.assertListEqual(list(gen_f), list(expected_f))

    def test_op_pymodel_builder(self):
        with TemporaryDirectory(dir='./') as gen_file_path:
            pyfile_name = 'resnet50_model.py'
            save_path = gen_file_path + '/' + pyfile_name
            gen_model_from_op(
                save_path=save_path,
                task_name='image_embedding',
                op_name='timm',
                op_init_args={'model_name': 'resnet50'},
                input_annotations=[(Image, (512, 512, 3))],
                output_annotations=[(np.float32, (1, 3, 224, 224))]
            )

            with open(save_path, 'rt', encoding='utf-8') as gen_f, \
                 open(EXPECTED_FILE_PATH + pyfile_name, 'rt', encoding='utf-8') as expected_f:
                a = list(gen_f)
                b = list(expected_f)
                for i in range(len(a)):
                    self.assertEqual(a[i].strip(), b[i].strip())


if __name__ == '__main__':
    unittest.main()
