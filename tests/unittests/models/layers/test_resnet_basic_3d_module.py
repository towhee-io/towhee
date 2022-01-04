# Copyright 2021 Zilliz and Facebook. All rights reserved.
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
import numpy as np
from torch import nn
import itertools


from towhee.models.layers.resnet_basic_3d_module import ResNetBasic3DModule
from towhee.models.utils.create_resnet_basic_3d_module import create_resnet_basic_3d_module


class TestResNetBasic3DModule(unittest.TestCase):
    """
    Test ResNet basic 3d module.
    """
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    @staticmethod
    def _get_inputs(input_dim: int = 3) -> torch.tensor:
        """
        Provide different tensors as test cases.
        Yield:
            (torch.tensor): tensor as test case input.
        """
        # Prepare random tensor as test cases.
        shapes = (
            # Forward succeeded.
            (1, input_dim, 3, 7, 7),
            (1, input_dim, 5, 7, 7),
            (1, input_dim, 7, 7, 7),
            (2, input_dim, 3, 7, 7),
            (4, input_dim, 3, 7, 7),
            (8, input_dim, 3, 7, 7),
            (2, input_dim, 3, 7, 14),
            (2, input_dim, 3, 14, 7),
            (2, input_dim, 3, 14, 14),
            # Forward failed.
            (8, input_dim * 2, 3, 7, 7),
            (8, input_dim * 4, 5, 7, 7),
        )
        for shape in shapes:
            yield torch.rand(shape)

    def test_create_simple_module(self):
        """
        Test simple ResNetBasicStem (without pooling layer).
        """
        for input_dim, output_dim in itertools.product((2, 3), (4, 8, 16)):
            model = ResNetBasic3DModule(
                conv=nn.Conv3d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                norm=nn.BatchNorm3d(output_dim),
                activation=nn.ReLU(),
                pool=None,
            )

            # Test forwarding.
            for tensor in TestResNetBasic3DModule._get_inputs(input_dim):
                if tensor.shape[1] != input_dim:
                    with self.assertRaises(RuntimeError):
                        output_tensor = model(tensor)
                    continue
                else:
                    output_tensor = model(tensor)

                input_shape = tensor.shape
                output_shape = output_tensor.shape
                output_shape_gt = (
                    input_shape[0],
                    output_dim,
                    input_shape[2],
                    input_shape[3],
                    input_shape[4],
                )

                self.assertEqual(
                    output_shape,
                    output_shape_gt,
                    f"Output shape {output_shape} is different from expected shape {output_shape_gt}"
                )

    def test_create_complex_module(self):
        """
        Test complex ResNetBasicStem.
        """
        for input_dim, output_dim in itertools.product((2, 3), (4, 8, 16)):
            model = ResNetBasic3DModule(
                conv=nn.Conv3d(
                    input_dim,
                    output_dim,
                    kernel_size=[3, 7, 7],
                    stride=[1, 2, 2],
                    padding=[1, 3, 3],
                    bias=False,
                ),
                norm=nn.BatchNorm3d(output_dim),
                activation=nn.ReLU(),
                pool=nn.MaxPool3d(
                    kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
                ),
            )

            # Test forwarding.
            for input_tensor in TestResNetBasic3DModule._get_inputs(input_dim):
                if input_tensor.shape[1] != input_dim:
                    with self.assertRaises(Exception):
                        output_tensor = model(input_tensor)
                    continue
                else:
                    output_tensor = model(input_tensor)

                input_shape = input_tensor.shape
                output_shape = output_tensor.shape

                output_shape_gt = (
                    input_shape[0],
                    output_dim,
                    input_shape[2],
                    (((input_shape[3] - 1) // 2 + 1) - 1) // 2 + 1,
                    (((input_shape[4] - 1) // 2 + 1) - 1) // 2 + 1,
                )

                self.assertEqual(
                    output_shape,
                    output_shape_gt,
                    f"Output shape {output_shape} is different from expected shape {output_shape_gt}"
                )

    def test_create_stem_with_callable(self):
        """
        Test builder `create_res_basic_stem` with callable inputs.
        """
        for (pool, activation, norm) in itertools.product(
            (nn.AvgPool3d, nn.MaxPool3d, None),
            (nn.ReLU, nn.Softmax, nn.Sigmoid, None),
            (nn.BatchNorm3d, None),
        ):
            model = create_resnet_basic_3d_module(
                in_channels=3,
                out_channels=64,
                pool=pool,
                activation=activation,
                norm=norm,
            )
            model_gt = ResNetBasic3DModule(
                conv=nn.Conv3d(
                    3,
                    64,
                    kernel_size=[3, 7, 7],
                    stride=[1, 2, 2],
                    padding=[1, 3, 3],
                    bias=False,
                ),
                norm=None if norm is None else norm(64),
                activation=None if activation is None else activation(),
                pool=None
                if pool is None
                else pool(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]),
            )

            model.load_state_dict(
                model_gt.state_dict(), strict=True
            )  # explicitly use strict mode.

            # Test forwarding.
            for input_tensor in TestResNetBasic3DModule._get_inputs():
                with torch.no_grad():
                    if input_tensor.shape[1] != 3:
                        with self.assertRaises(RuntimeError):
                            output_tensor = model(input_tensor)
                        continue
                    else:
                        output_tensor = model(input_tensor)
                        output_tensor_gt = model_gt(input_tensor)
                self.assertEqual(
                    output_tensor.shape,
                    output_tensor_gt.shape,
                    f"Output shape {output_tensor.shape} is different from expected shape {output_tensor_gt.shape}"
                )
                self.assertTrue(
                    np.allclose(output_tensor.numpy(), output_tensor_gt.numpy())
                )
