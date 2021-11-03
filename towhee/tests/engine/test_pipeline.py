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

# import os
from pathlib import Path
import unittest
from shutil import rmtree


from PIL import Image

from towhee import pipeline
from towhee.engine.engine import EngineConfig
from towhee.hub.file_manager import FileManagerConfig


CACHE_PATH = Path(__file__).parent.parent.resolve()


class TestPipeline(unittest.TestCase):
    """
    Tests `pipeline` functionality.
    """
    def setUp(self):
        conf = EngineConfig()
        conf.cache_path = CACHE_PATH
        conf.sched_interval_ms = 20
        new_cache = (CACHE_PATH/'test_cache')
        pipeline_cache = (CACHE_PATH/'test_util')
        operator_cache = (CACHE_PATH/'mock_operators')
        fmc = FileManagerConfig()
        fmc.change_default_cache(new_cache)
        pipelines = [path for path in (pipeline_cache).rglob('*.yaml')]
        operators = [f for f in operator_cache.iterdir() if f.is_dir()]
        fmc.cache_local_pipeline(pipelines)
        fmc.cache_local_operator(operators)
        
        


    def test_empty_input(self):
        p = pipeline('local/simple_pipeline', fmc=FileManagerConfig())
        self.assertEqual(p(), [])

    def test_simple_pipeline(self):
        p = pipeline('local/simple_pipeline', fmc=FileManagerConfig())
        res = p(0)
        self.assertEqual(res[0], 3)

    def test_embedding_pipeline(self):
        p = pipeline('local/resnet50_embedding',
                     fmc=FileManagerConfig())
        img_path = CACHE_PATH / 'data' / 'dataset' / 'kaggle_dataset_small' / \
            'train' / '0021f9ceb3235effd7fcde7f7538ed62.jpg'
        img = Image.open(str(img_path))
        res = p(img)
        self.assertEqual(res[0].size, 1000)

    def test_simple_pipeline_multirow(self):
        #pylint: disable=protected-access
        p = pipeline('local/simple_pipeline', fmc=FileManagerConfig())
        p._pipeline.parallelism = 2
        res = p(list(range(1000)))
        for n in range(1000):
            self.assertEqual(res[n], n+3)



# class TestPipelineCache(unittest.TestCase):
#     def test_pipeline_cache(self):
#         self.assertEqual(_get_pipeline_cache(
#             None), Path.home() / '.towhee/pipelines')

#         os.environ[_PIPELINE_CACHE_ENV] = '/opt/.pipeline'
#         self.assertEqual(_get_pipeline_cache(
#             None), Path('/opt/.pipeline'))

#         self.assertEqual(_get_pipeline_cache(
#             '/home/mycache'), Path('/home/mycache'))


if __name__ == '__main__':
    unittest.main()
    # x = TestPipeline()
    # x.setUp()
    