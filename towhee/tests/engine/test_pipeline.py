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

import os
from pathlib import Path
import unittest


from PIL import Image

from towhee import pipeline, _get_pipeline_cache, _PIPELINE_CACHE_ENV
from towhee.tests import CACHE_PATH


class TestPipeline(unittest.TestCase):
    """
    Tests `pipeline` functionality.
    """

    # def test_empty_input(self):
    #     p = pipeline('test_util/simple_pipeline', cache=str(CACHE_PATH))
    #     self.assertEqual(p(), [])

    # def test_simple_pipeline(self):
    #     p = pipeline('test_util/simple_pipeline', cache=str(CACHE_PATH))
    #     res = p(0)
    #     self.assertEqual(res[0], 3)

    # def test_embedding_pipeline(self):
    #     p = pipeline('test_util/resnet50_embedding',
    #                  cache=str(CACHE_PATH))
    #     img_path = CACHE_PATH / 'data' / 'dataset' / 'kaggle_dataset_small' / \
    #         'train' / '0021f9ceb3235effd7fcde7f7538ed62.jpg'
    #     img = Image.open(str(img_path))
    #     res = p(img)
    #     self.assertEqual(res[0].size, 1000)

    # def test_simple_pipeline_multirow(self):
    #     #pylint: disable=protected-access
    #     p = pipeline('test_util/simple_pipeline', cache=str(CACHE_PATH))
    #     p._pipeline.parallelism = 2
    #     res = p(list(range(1000)))
    #     for n in range(1000):
    #         self.assertEqual(res[n], n+3)

    def test_concat(self):
        #pylint: disable=protected-access
        p = pipeline('test_util/test_concat', cache=str(CACHE_PATH))
        res1 = p([(0,1,2), (3,4,5), (6,7,8)])
        print(res1)
        self.assertEqual(res1, [(5, 7, 9), (8, 10, 12), (11, 13, 15)])
        res2 = p([(9,10,11), (12,13,14), ])
        self.assertEqual(res2, [(14, 16, 18), (17, 19, 21)])

    def test_multi_column_output(self):
        p = pipeline('test_util/test_multi_column_out', cache=str(CACHE_PATH))
        print( p([(0,1,2), (3,4,5), (6,7,8)]))



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
