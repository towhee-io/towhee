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

from towhee.models import acar_net


class TestNeck(unittest.TestCase):
    """
    Test ACAR-NET Neck
    """
    def test_basic(self):
        data = {
            # 'clips': [torch.rand(1, 3, 10, 224, 224)],
            'aug_info': [{'crop_box': [0.0, 0.0, 1.0, 1.0], 'flip': False, 'pad_ratio': [1.0, 1.0]}],
            'filenames': ['dummy video'],
            'labels': [[
                {'label': [0], 'bounding_box': [0.0, 0.0, 3.0, 4.0]},
                {'label': [1], 'bounding_box': [10.0, 10.0, 20.0, 20.0]}
            ]],
            'mid_times': [5]
        }
        neck = acar_net.neck('basic', num_classes=3)
        out = neck(data)
        self.assertTrue(out['filenames'] == ['dummy video'])
        self.assertTrue(out['mid_times'] == [5])
        self.assertTrue(out['num_rois'] == 1)
        self.assertTrue((out['rois'] == torch.tensor([[0, 0, 0, 1, 1]])).all())
        self.assertTrue(out['roi_ids'] == [0, 1])
        self.assertTrue((out['targets'] == torch.tensor([[1, 0, 0]])).all())
        self.assertTrue(out['sizes_before_padding'] == [[1.0, 1.0]])
        self.assertTrue(out['bboxes'] == [[0.0, 0.0, 3.0, 4.0]])
        self.assertTrue(out['bbox_ids'] == [0])


if __name__ == '__main__':
    unittest.main()
