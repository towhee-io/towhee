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

import unittest
import math
from urllib import request
from towhee.trainer.models.vit.vit_pretrained import PRETRAINED_MODELS


class VitPretrainedTest(unittest.TestCase):
    name = 'B_16_imagenet1k'

    def test_pretrained(self):
        assert self.name in PRETRAINED_MODELS.keys(), \
            'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
        with request.urlopen(PRETRAINED_MODELS[self.name]['url']) as file:
            self.assertEqual(math.floor(file.status/200), 2)
            self.assertEqual(file.reason, 'OK')


if __name__ == '__main__':
    unittest.main()
