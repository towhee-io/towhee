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
import shutil
from pathlib import Path

from towhee.pipelines import image_embedding_pipeline
from towhee import ops

cache_path = Path(__file__).parent.parent.resolve() / 'test_cache' / 'pipelines' / 'image_embedding_test'


class TestImageEmbeddingPipeline(unittest.TestCase):
    """
    tests for template build
    """
    def test_image_embedding_pipeline(self):
        if cache_path.exists():
            shutil.rmtree(cache_path)

        pipeline = image_embedding_pipeline('efficientnet-b3')
        pipeline.save(name='test_image_embedding_pipeline', path=cache_path)
        self.assertTrue((cache_path / 'test_image_embedding_pipeline').exists())

        pipeline = image_embedding_pipeline(['efficientnet-b3', ops.filip_halt.timm_image_embedding(model_name='regnety-004')])
        pipeline.save(name='test_image_embedding_pipeline_2way', path=cache_path)
        self.assertTrue((cache_path / 'test_image_embedding_pipeline_2way').exists())

        pipeline = image_embedding_pipeline(['efficientnet-b3', 'efficientnet-b3', ops.filip_halt.timm_image_embedding(model_name='regnety-004')])
        pipeline.save(name='test_image_embedding_pipeline_3way', path=cache_path)
        self.assertTrue((cache_path / 'test_image_embedding_pipeline_3way').exists())


if __name__ == '__main__':
    unittest.main()
