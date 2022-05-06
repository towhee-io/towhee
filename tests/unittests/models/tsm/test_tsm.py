# Copyright 2022 Zilliz. All rights reserved.
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

import torch
import unittest
from towhee.models.tsm.tsm import TSN

class TSMTest(unittest.TestCase):
    def test_tsm_resnet_rgb(self):
        num_class = 256
        num_segments = 1
        modality = 'RGB'
        is_shift = True
        non_local = True
        input_tensor=torch.randn(10,3,224,224)
        model = TSN(num_class = num_class, num_segments = num_segments, modality = modality, is_shift = is_shift, non_local = non_local)
        out = model(input_tensor)
        self.assertTrue(out.shape == torch.Size([10,256]))

    def test_tsm_resnet_flow(self):
        num_class = 256
        num_segments = 1
        modality = 'Flow'
        is_shift = True
        non_local = True
        input_tensor=torch.randn(10,3,224,224)
        model = TSN(num_class = num_class, num_segments = num_segments, modality = modality, is_shift = is_shift, non_local = non_local)
        out = model(input_tensor)
        self.assertTrue(out.shape == torch.Size([3,256]))

    def test_tsm_resnet_diff(self):
        num_class = 256
        num_segments = 3
        modality = 'RGBDiff'
        is_shift = True
        non_local = True
        input_tensor=torch.randn(54,3,224,224)
        model = TSN(num_class = num_class, num_segments = num_segments, modality = modality, is_shift = is_shift, non_local = non_local)
        out = model(input_tensor)
        self.assertTrue(out.shape == torch.Size([3,256]))

    def test_tsm_mobilenetv2_rgb(self):
        num_class = 256
        num_segments = 1
        modality = 'RGB'
        base_model = 'mobilenetv2'
        is_shift = True
        non_local = True
        input_tensor=torch.randn(10,3,224,224)
        model = TSN(num_class = num_class,
                    num_segments = num_segments,
                    modality = modality,
                    base_model = base_model,
                    is_shift = is_shift,
                    non_local = non_local)
        out = model(input_tensor)
        self.assertTrue(out.shape == torch.Size([10,256]))

    def test_tsm_mobilenetv2_flow(self):
        num_class = 256
        num_segments = 1
        modality = 'Flow'
        base_model = 'mobilenetv2'
        is_shift = True
        non_local = True
        input_tensor=torch.randn(10,3,224,224)
        model = TSN(num_class = num_class,
                    num_segments = num_segments,
                    modality = modality,
                    base_model = base_model,
                    is_shift = is_shift,
                    non_local = non_local)
        out = model(input_tensor)
        self.assertTrue(out.shape == torch.Size([3,256]))

    def test_tsm_mobilenetv2_diff(self):
        num_class = 256
        num_segments = 3
        modality = 'RGBDiff'
        base_model = 'mobilenetv2'
        is_shift = True
        non_local = True
        input_tensor=torch.randn(54,3,224,224)
        model = TSN(num_class = num_class,
                    num_segments = num_segments,
                    modality = modality,
                    base_model = base_model,
                    is_shift = is_shift,
                    non_local = non_local)
        out = model(input_tensor)
        self.assertTrue(out.shape == torch.Size([3,256]))
