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

from towhee.models.acar_net import AcarNet


class TestNeck(unittest.TestCase):
    """
    Test ACAR-NET model
    """
    data = {
        'clips': [torch.rand(1, 3, 8, 64, 64)],
        'aug_info': [[{'crop_box': [0.0, 0.0, 1.0, 1.0], 'flip': False, 'pad_ratio': [1.0, 1.0]}]],
        'filenames': ['dummy video'],
        'labels': [[
            {'label': [0], 'bounding_box': [0.0, 0.0, 3.0, 4.0]},
            {'label': [1], 'bounding_box': [10.0, 10.0, 20.0, 20.0]}
        ]],
        'mid_times': [5]
    }

    def test_linear(self):
        configs = {
            'backbone': {
                'model_name': 'slowfast50',
                'alpha': 4,
                'beta': 0.125,
                'fuse_only_conv': False,
                'fuse_kernel_size': 7,
                'slow_full_span': True
            },
            'neck': {
                'model_name': 'basic',
                'num_classes': 3,
                'aug_threshold': 0.,
                'multi_class': True
            },
            'head': {
                'model_name': 'linear',
                'width': 2304,
                'num_classes': 3
            }
        }
        model = AcarNet(configs)
        out = model(self.data)

        self.assertTrue(out['outputs'].shape == (1, 3))
        self.assertTrue(out['filenames'] == ['dummy video'])
        self.assertTrue(out['mid_times'] == [5])
        self.assertTrue(out['num_rois'] == 1)
        self.assertTrue((out['targets'] == torch.tensor([[1, 0, 0]])).all())
        self.assertTrue(out['bboxes'] == [[0.0, 0.0, 3.0, 4.0]])

    def test_acar(self):
        configs = {
            'backbone': {
                'model_name': 'slowfast101',
                'alpha': 4,
                'beta': 0.125,
            },
            'neck': {
                'model_name': 'basic',
                'num_classes': 3,
            },
            'head': {
                'model_name': 'acar',
                'width': 2304,
                'num_classes': 3
            }
        }
        model = AcarNet(configs)
        out = model(self.data)

        self.assertTrue(out['outputs'].shape == (1, 3))
        self.assertTrue(out['filenames'] == ['dummy video'])
        self.assertTrue(out['mid_times'] == [5])
        self.assertTrue(out['num_rois'] == 1)
        self.assertTrue((out['targets'] == torch.tensor([[1, 0, 0]])).all())
        self.assertTrue(out['bboxes'] == [[0.0, 0.0, 3.0, 4.0]])


if __name__ == '__main__':
    unittest.main()
