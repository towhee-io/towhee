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
from towhee.models.layers.padding_functions import get_padding, get_same_padding, is_static_pad, pad_same, \
    get_padding_value, same_padding


class TestPaddingFunctions(unittest.TestCase):
    """
    Test Padding utility functions
    """

    def test_get_padding(self):
        """
        Test get_padding_function.
        """
        kernel_size = 3
        stride = 1
        dilation = 1
        self.assertEqual(get_padding(kernel_size, stride, dilation), 1)

        kernel_size = 3
        stride = 2
        dilation = 2
        self.assertEqual(get_padding(kernel_size, stride, dilation), 2)

    def test_get_same_padding(self):
        """
        Test get_same_padding function.
        """
        x = 224
        k = 3
        s = 1
        d = 1
        self.assertEqual(get_same_padding(x, k, s, d), 2)

        x = 224
        k = 5
        s = 1
        d = 1
        self.assertEqual(get_same_padding(x, k, s, d), 4)

    def test_is_static_pad(self):
        """
        Test get_same_padding function.
        """
        kernel = 3
        stride = 1
        dilation = 1
        self.assertEqual(is_static_pad(kernel, stride, dilation), True)

        kernel = 3
        stride = 1
        dilation = 2
        self.assertEqual(is_static_pad(kernel, stride, dilation), True)

    def test_get_padding_value(self):
        """
        Test get_padding_value function.
        """
        kernel = 3
        stride = 1
        dilation = 1

        padding, dynamic = get_padding_value('same', kernel, stride=stride, dilation=dilation)
        self.assertEqual(padding, 1)
        self.assertEqual(dynamic, False)

    def test_pad_same(self):
        """
        Test pad same function.
        """
        x = torch.Tensor(1, 3, 4, 4)
        k = [3, 3]
        s = [1, 1]
        d = [1, 1]
        value = 128.0
        out = pad_same(x, k, s, d, value)
        self.assertEqual(out.shape, torch.Size((1, 3, 6, 6)))
        self.assertEqual(out[0, 0, 0, 0], 128.0)

    def test_same_padding(self):
        """
        Test same padding function.
        """
        in_height = 4
        in_width = 4
        stride_h = 1
        stride_w = 1
        filter_height = 1
        filter_width = 1
        x = torch.Tensor(1, 3, 4, 4)
        out = same_padding(x = x,
                    in_height = in_height,
                    in_width = in_width,
                    stride_h = stride_h,
                    stride_w = stride_w,
                    filter_height = filter_height,
                    filter_width = filter_width,
                    )
        print(out.shape)
        self.assertTrue(out.shape == torch.Size([1, 3, 4, 4]))

if __name__ == '__main__':
    unittest.main()
