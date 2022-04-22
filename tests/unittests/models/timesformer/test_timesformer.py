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
from towhee.models.timesformer.timesformer import TimeSformer


class TestTimesformer(unittest.TestCase):
    def test_timesformer(self):
        for attention_type in ['divided_space_time', 'space_only', 'joint_space_time']:
            model = TimeSformer(img_size=224, num_classes=400, num_frames=4, attention_type=attention_type)
            dummy_video = torch.randn(1, 3, 4, 224, 224)  # (batch x channels x frames x height x width)
            pred = model(dummy_video)
            self.assertTrue(pred.shape == (1, 400))


if __name__ == '__main__':
    unittest.main()
