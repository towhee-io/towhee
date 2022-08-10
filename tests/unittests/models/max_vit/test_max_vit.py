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
import torch
from towhee.models.max_vit.max_vit import create_model


class MaxVitUtilsTest(unittest.TestCase):

    def test_max_vit(self):

        data = torch.rand(1, 3, 224, 224)
        if torch.cuda.is_available():
            data = data.cuda()

        model = create_model(model_name='max_vit_tiny')
        output = model(data)
        self.assertEqual(output.shape, (1, 1000))


if __name__ == '__main__':
    unittest.main()
