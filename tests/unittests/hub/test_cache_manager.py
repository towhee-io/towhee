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

from towhee.hub.cache_manager import CacheManager


class TestCacheManager(unittest.TestCase):
    """
    Simple hub download and run test.
    """
    def test_error_repo(self):
        with self.assertRaises(RuntimeError):
            CacheManager().get_operator('No-user/No-op', 'main', True, False)


    def test_download_op(self):
        CacheManager().get_operator('image-decode/cv2-rgb', 'main', True, True)
        CacheManager().get_operator('image-decode/cv2', 'main', True, False)
