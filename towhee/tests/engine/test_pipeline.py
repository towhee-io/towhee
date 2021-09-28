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


from pathlib import Path
import unittest

from PIL import Image

from towhee.engine import pipeline
from towhee.engine import Engine
from towhee.engine import EngineConfig


CACHE_PATH = Path(__file__).parent.parent.resolve()


class TestPipeline(unittest.TestCase):
    """Tests `pipeline` functionality.
    """

    def setUp(self):
        conf = EngineConfig()
        conf.cache_path = CACHE_PATH
        conf.sched_interval_ms = 20
        engine = Engine()
        if not engine.is_alive():
            engine.start()

    def test_simple_pipeline(self):
        p = pipeline('test_util/simple_pipeline', cache_path=str(CACHE_PATH))
        res = p(0)
        self.assertEqual(res[0], 3)

    def test_embedding_pipeline(self):
        p = pipeline('test_util/resnet50_embedding', cache_path=str(CACHE_PATH))
        img_path = CACHE_PATH / 'dataset' / 'kaggle_dataset_small' / 'train' / '0021f9ceb3235effd7fcde7f7538ed62.jpg'
        img = Image.open(str(img_path))
        res = p(img)
        print('successfully got an embedding of size {0}'.format(res[0].size))
        self.assertEqual(res[0].size, 1000)


if __name__ == '__main__':
    unittest.main()
