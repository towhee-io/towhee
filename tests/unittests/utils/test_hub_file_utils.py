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

from towhee.utils.hub_file_utils import HubFileUtils

hub_file = HubFileUtils()


class TestHubUtils(unittest.TestCase):
    """
    Unittest for hub utils.
    """
    def test_token(self):
        hub_file.set_token('test-token')
        hub_file.save()
        token1 = hub_file.get_token()
        self.assertEqual(token1, 'test-token')
        hub_file.delete()
        token2 = hub_file.get_token()
        self.assertEqual(token2, None)


if __name__ == '__main__':
    unittest.main()
