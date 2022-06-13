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
import unittest

from typing import List
import numpy as np

from serve.triton.type_gen import gen_from_type_annotations
from towhee.types import VideoFrame, AudioFrame


class TestGenFromTypeAnnotations(unittest.TestCase):
    """
    Unit test for gen_from_type_annotations.
    """

    def test_type_checking(self):
        funcs = {
            np.ndarray: lambda t, s: 'ndarray',
            str: lambda t, s: 'str',
            int: lambda t, s: 'int',
            VideoFrame: lambda t, s: 'VideoFrame',
            AudioFrame: lambda t, s: 'AudioFrame',
        }

        annotations = [
            (np.ndarray, (4, 4)),
            (str, (-1)),
            (int, ()),
            (VideoFrame, (-1, 512, 512)),
            (AudioFrame, (-1, 512, 512)),
            (List[np.ndarray], (-1, 512, 512)),
        ]

        expected_results = [
            'ndarray',
            'str',
            'int',
            'VideoFrame',
            'AudioFrame',
            'ndarray',
        ]

        results = gen_from_type_annotations(annotations, funcs)

        self.assertListEqual(expected_results, results)


if __name__ == '__main__':
    unittest.main()
