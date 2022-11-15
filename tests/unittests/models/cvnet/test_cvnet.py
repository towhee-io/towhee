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
from towhee.models import cvnet


class TestCVNet(unittest.TestCase):
    """
    Test svt model
    """
    query_image = torch.randn(1, 3, 24, 24)
    key_image = torch.randn(1, 3, 24, 24)

    def test_cvnet_r101(self):
        if torch.cuda.is_available() is True:
            self.query_image = self.query_image.cuda()
            self.key_image = self.key_image.cuda()
        model = cvnet.create_model(
            model_name='CVNet_R101',
            pretrained=False, resnet_depth=101, reduction_dim=2048)
        score = model(self.query_image, self.key_image).unsqueeze(-1)
        self.assertTrue(score.shape == (1, 1))
        global_descriptor = model.extract_global_descriptor(self.query_image)
        self.assertTrue(global_descriptor.shape == (1, 2048))
        query_feature = model.extract_featuremap(self.query_image)
        key_feature = model.extract_featuremap(self.key_image)
        self.assertTrue(query_feature[0].shape == (1, 1024, 2, 2))
        self.assertTrue(key_feature[0].shape == (1, 1024, 2, 2))
        final_score = model.extract_score_with_featuremap(query_feature, key_feature).unsqueeze(-1).unsqueeze(-1)
        self.assertTrue(final_score.shape == (1, 1))

    def test_cvnet_r50(self):
        if torch.cuda.is_available() is True:
            self.query_image = self.query_image.cuda()
            self.key_image = self.key_image.cuda()
        model = cvnet.create_model(
            model_name='CVNet_R50',
            pretrained=False, resnet_depth=50, reduction_dim=2048)
        score = model(self.query_image, self.key_image).unsqueeze(-1)
        self.assertTrue(score.shape == (1, 1))
        global_descriptor = model.extract_global_descriptor(self.query_image)
        self.assertTrue(global_descriptor.shape == (1, 2048))
        query_feature = model.extract_featuremap(self.query_image)
        key_feature = model.extract_featuremap(self.key_image)
        self.assertTrue(query_feature[0].shape == (1, 1024, 2, 2))
        self.assertTrue(key_feature[0].shape == (1, 1024, 2, 2))
        final_score = model.extract_score_with_featuremap(query_feature, key_feature).unsqueeze(-1).unsqueeze(-1)
        self.assertTrue(final_score.shape == (1, 1))


if __name__ == '__main__':
    unittest.main()
