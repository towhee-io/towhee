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
import os
from towhee.models.utils.download import download_from_url, checksum


class TestDownload(unittest.TestCase):
    """
    Test download tools
    """
    def setUp(self):
        self.url = 'https://raw.githubusercontent.com/towhee-io/towhee/main/towhee_logo.png'
        self.target = './'
        self.target_file = os.path.join(self.target, 'towhee_logo.png')

    def test_download1(self):
        download_from_url(self.url, './', hash_prefix=None)
        self.assertTrue(os.path.isfile(self.target_file))

        download_from_url(self.url, './', hash_prefix='7222f2e')
        self.assertTrue(os.path.isfile(self.target_file))

    def test_download2(self):
        download_from_url(self.url, './', hash_prefix='7222f2e')
        self.assertTrue(os.path.isfile(self.target_file))

        download_from_url(self.url, './', hash_prefix=None)
        self.assertTrue(os.path.isfile(self.target_file))

    def test_error(self):
        try:
            os.makedirs(self.target_file, exist_ok=True)
            download_from_url(self.url, './')
        except RuntimeError:
            os.removedirs(self.target_file)
        try:
            download_from_url(self.url, './', hash_prefix='xxx')
        except AssertionError:
            checksum(self.target_file, '7222f2e')

        download_from_url(self.url, './', hash_prefix='7222f2e')
        self.assertTrue(os.path.isfile(self.target_file))

    def tearDown(self):
        if os.path.exists(self.target_file):
            os.remove(self.target_file)


if __name__ == '__main__':
    unittest.main()
