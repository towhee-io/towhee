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
from pathlib import Path
# from shutil import rmtree

from towhee import pipeline
from tests.unittests import CACHE_PATH

cache_path = Path(__file__).parent.parent.resolve()


# @unittest.skip('Not pass')
class TestDownload(unittest.TestCase):
    """
    Simple hub download and run test.
    """

    def test_pipeline(self):
        p = pipeline('image-embedding')
        img = str(CACHE_PATH / 'mock_pipelines/ci_test/towhee_logo.png')
        res = p(img)
        self.assertIsInstance(res, np.ndarray)

    def test_tag(self):
        p = pipeline('towhee/ci-test')
        res = p('test')

        self.assertIn('test on main', res[0])

        p = pipeline('towhee/ci-test', tag='test')
        res = p('test')

        self.assertIn('test on test', res[0])


if __name__ == '__main__':
    unittest.main()
