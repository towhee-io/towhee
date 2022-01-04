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

import torch
import unittest

from towhee.models.multiscale_vision_transformers.create_multiscale_vision_transformers \
    import create_multiscale_vision_transformers


class MultiscaleVisionTransformersTest(unittest.TestCase):
    def test_multiscale_vision_transformers(self):
        # Test MViT with 3D case.
        num_head = 100
        batch_size = 1
        fake_input = torch.rand(batch_size, 3, 4, 28, 28)
        model = create_multiscale_vision_transformers(
            spatial_size=28,
            temporal_size=4,
            patch_embed_dim=12,
            depth=1,
            head_num_classes=num_head,
            pool_kv_stride_adaptive=[1, 2, 2],
        )
        output = model(fake_input)
        gt_shape_tensor = torch.rand(batch_size, num_head)
        self.assertTrue(output.shape == gt_shape_tensor.shape)
        # Test MViT with 3D case with pool first.
        num_head = 100
        batch_size = 1
        fake_input = torch.rand(batch_size, 3, 4, 28, 28)
        model = create_multiscale_vision_transformers(
            spatial_size=28,
            temporal_size=4,
            patch_embed_dim=12,
            depth=1,
            head_num_classes=num_head,
            pool_first=True,
            pool_q_stride_size=[[0, 1, 2, 2]],
        )
        output = model(fake_input)
        gt_shape_tensor = torch.rand(batch_size, num_head)
        self.assertTrue(output.shape == gt_shape_tensor.shape)

        # Test MViT with 2D case for images.
        conv_patch_kernel = (7, 7)
        conv_patch_stride = (4, 4)
        conv_patch_padding = (3, 3)
        num_head = 100
        batch_size = 1
        fake_input = torch.rand(batch_size, 3, 28, 28)
        model = create_multiscale_vision_transformers(
            spatial_size=(28, 28),
            temporal_size=1,
            patch_embed_dim=12,
            depth=1,
            head_num_classes=num_head,
            use_2d_patch=True,
            conv_patch_embed_kernel=conv_patch_kernel,
            conv_patch_embed_stride=conv_patch_stride,
            conv_patch_embed_padding=conv_patch_padding,
        )
        output = model(fake_input)
        gt_shape_tensor = torch.rand(batch_size, num_head)
        self.assertTrue(output.shape == gt_shape_tensor.shape)

        # Test MViT without patch_embed.
        num_head = 100
        batch_size = 1
        fake_input = torch.rand(batch_size, 8, 12)
        model = create_multiscale_vision_transformers(
            spatial_size=(8, 1),
            temporal_size=1,
            patch_embed_dim=12,
            depth=1,
            enable_patch_embed=False,
            head_num_classes=num_head,
        )
        output = model(fake_input)
        gt_shape_tensor = torch.rand(batch_size, num_head)
        self.assertTrue(output.shape == gt_shape_tensor.shape)


if __name__ == '__main__':
    unittest.main()
