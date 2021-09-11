# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# import os
# import tempfile
import unittest
import torch


class ScheduleInitTest(unittest.TestCase):

    def test_schedulers(self):
        str1 = torch.__version__
        self.assertEqual(True,len(str1) > 0)


if __name__ == '__main__':
    unittest.main()
