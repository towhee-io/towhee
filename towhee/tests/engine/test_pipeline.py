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
from towhee.tests import CACHE_PATH
from towhee.hub.file_manager import FileManagerConfig, FileManager


class TestPipeline(unittest.TestCase):
    """
    Tests `pipeline` functionality.
    """
    @classmethod
    def setUpClass(cls):
        new_cache = (CACHE_PATH/'test_cache')
        pipeline_cache = (CACHE_PATH/'test_util')
        operator_cache = (CACHE_PATH/'mock_operators')
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

    def test_embedding_pipeline(self):
        p = pipeline('local/resnet50_embedding')
        img_path = CACHE_PATH / 'data' / 'dataset' / 'kaggle_dataset_small' / \
            'train' / '0021f9ceb3235effd7fcde7f7538ed62.jpg'
        img = Image.open(str(img_path))
        res = p(img)
        self.assertEqual(res[0][0].size, 1000)

    def test_error_input(self):
        p = pipeline('local/simple_pipeline')
        with self.assertRaises(RuntimeError):
            p('xx')

    # def test_simple_pipeline_multirow(self):
    #     #pylint: disable=protected-access
    #     p = pipeline('test_util/simple_pipeline', cache=str(CACHE_PATH))
    #     p._pipeline.parallelism = 2
    #     res = p(list(range(1000)))
    #     for n in range(1000):
    #         self.assertEqual(res[n], n+3)


# class TestPipelineCache(unittest.TestCase):
#     def test_pipeline_cache(self):
#         self.assertEqual(_get_pipeline_cache(
#             None), Path.home() / '.towhee/pipelines')

#         os.environ[_PIPELINE_CACHE_ENV] = '/opt/.pipeline'
#         self.assertEqual(_get_pipeline_cache(
#             None), Path('/opt/.pipeline'))

if __name__ == '__main__':
    unittest.main()
