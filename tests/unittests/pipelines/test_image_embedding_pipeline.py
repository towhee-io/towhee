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

from towhee.pipelines import image_embedding_pipeline
from towhee import ops


class TestImageEmbeddingPipeline(unittest.TestCase):
    """
    tests for template build
    """

    def test_image_embedding_pipeline(self):
        pipeline = image_embedding_pipeline('efficientnet-b3')
        pipeline.save('test_image_embedding_pipeline')

        pipeline = image_embedding_pipeline([
            'efficientnet-b3',
            ops.filip_halt.timm_image_embedding(model_name='regnety-004')
        ])
        pipeline.save('test_image_embedding_pipeline_2way')

        pipeline = image_embedding_pipeline([
            'efficientnet-b3', 'efficientnet-b3',
            ops.filip_halt.timm_image_embedding(model_name='regnety-004')
        ])
        pipeline.save('test_image_embedding_pipeline_3way')


if __name__ == '__main__':
    unittest.main()
