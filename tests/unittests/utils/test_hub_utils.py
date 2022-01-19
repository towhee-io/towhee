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

from towhee.utils.hub_utils import HubUtils


class TestHubUtils(unittest.TestCase):
    """
    Unittest for hub utils.
    """
    def test_convert_dict(self):
        d = {'test': '<class \'torch.Tensor\'>'}
        d = HubUtils.convert_dict(d)
        self.assertEqual(d['test'], 'torch.Tensor')

    def test_update_text(self):
        with open('text.txt', 'w', encoding='utf-8') as f1:
            f1.write('Hello towhee in test_ori.')
        HubUtils.update_text(['Hello', 'test_ori'], ['Unittest', 'test_tar'], 'text.txt', 'new_text.txt')
        with open('new_text.txt', 'r', encoding='utf-8') as f1:
            new_text = f1.read()
        assert 'test_tar' in new_text and 'Unittest' in new_text
        os.remove('text.txt')
        os.remove('new_text.txt')


if __name__ == '__main__':
    unittest.main()
