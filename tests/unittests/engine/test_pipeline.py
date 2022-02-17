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
from PIL import Image
# from shutil import rmtree

from towhee import pipeline
from towhee.errors import OpFailedError
from towhee.hub.file_manager import FileManagerConfig, FileManager

from tests.unittests import CACHE_PATH


class TestPipeline(unittest.TestCase):
    """
    Tests `pipeline` functionality.
    """
    @classmethod
    def setUpClass(cls):
        new_cache = (CACHE_PATH / 'test_cache')
        pipeline_cache = (CACHE_PATH / 'test_util')
        operator_cache = (CACHE_PATH / 'mock_operators')
        fmc = FileManagerConfig()
        fmc.update_default_cache(new_cache)
        pipelines = list(pipeline_cache.rglob('*.yaml'))
        operators = [f for f in operator_cache.iterdir() if f.is_dir()]
        fmc.cache_local_pipeline(pipelines)
        fmc.cache_local_operator(operators)
        FileManager(fmc)

    def test_empty_input(self):
        p = pipeline('local/simple_pipeline')
        with self.assertRaises(RuntimeError):
            p()

    def test_simple_pipeline(self):
        p = pipeline('local/simple_pipeline')
        res = p(0)
        self.assertEqual(res[0][0], 3)

    def test_flatmap_pipeline(self):
        p = pipeline('local/flatmap_pipeline')
        res = p(10)
        self.assertEqual(len(res), 5)

    def test_filter_pipeline(self):
        p = pipeline('local/filter_pipeline')
        res = p([1, 0, 2, 0, 3, 4])
        num = 1
        self.assertEqual(len(res), 4)
        for item in res:
            self.assertEqual(item[0], num)
            num += 1

    def test_embedding_pipeline(self):
        p = pipeline('local/resnet50_embedding')
        img_path = CACHE_PATH / 'data' / 'dataset' / 'kaggle_dataset_small' / \
            'train' / '0021f9ceb3235effd7fcde7f7538ed62.jpg'
        img = Image.open(str(img_path))
        res = p(img)
        self.assertEqual(res[0][0].size, 1000)

    def test_embedding_pipeline_with_format(self):
        p = pipeline('local/resnet50_embedding_with_format')
        img_path = CACHE_PATH / 'data' / 'dataset' / 'kaggle_dataset_small' / \
            'train' / '0021f9ceb3235effd7fcde7f7538ed62.jpg'
        img = Image.open(str(img_path))
        res = p(img)
        self.assertEqual(res.size, 1000)

    def test_error_input(self):
        p = pipeline('local/simple_pipeline')
        with self.assertRaises(OpFailedError):
            p('xx')

    def test_concat(self):
        p = pipeline('local/test_concat')
        res = p(0, 0, 0)
        self.assertEqual(res, [(1, 0, 2, 0, 3, 0)])

    def test_window(self):
        p = pipeline('local/test_window')
        res = p(1)
        self.assertEqual(len(res), 20)
        for i in range(20):
            if i < 18:
                self.assertEqual(res[i][0], 4)
            else:
                self.assertEqual(res[i][0], 2)

    def test_generator(self):
        p = pipeline('local/test_generator')
        input_data = 10
        res = p(input_data)
        self.assertEqual(len(res), input_data)
        for i in range(input_data):
            self.assertEqual(res[i][0], i)

        input_data = 20
        res = p(input_data)
        self.assertEqual(len(res), input_data)
        for i in range(input_data):
            self.assertEqual(res[i][0], i)

    def test_time_window(self):
        '''
        timestamp (ms)
            [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
        data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        window: (2, 3)
        windows: [0, 1, 2, 3], [6, 7, 8, 9]
        result: [6, 6, 30, 30]
        '''
        p = pipeline('local/test_timestamp')
        input_data = 10
        res = p(input_data)
        self.assertEqual([item[0] for item in res], [6, 6, 30, 30])


if __name__ == '__main__':
    unittest.main()
