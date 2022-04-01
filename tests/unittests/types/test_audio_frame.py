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

from towhee.types import AudioFrame
import numpy as np


class TestAudioFrame(unittest.TestCase):
    """
    tests for AudioFrame
    """

    def test_ndarray_methods(self):
        # pylint: disable=protected-access
        data = np.array([0, 0.707, 1, 0.707, 0, -0.707, -1, -0.707, 0])
        frame = AudioFrame(data, sample_rate=11025, timestamp=1)
        self.assertEqual(frame.sample_rate, 11025)
        self.assertEqual(frame.timestamp, 1)
        self.assertListEqual(list(frame.shape), [9])
        self.assertEqual(frame.sum(), 0)

        arr = np.zeros(9)
        frame_arr = arr.view(AudioFrame)
        self.assertListEqual(list(frame_arr.shape), [9])
        self.assertEqual(frame_arr.sum(), 0)
        self.assertEqual(frame_arr.sample_rate, None)
        self.assertEqual(frame_arr.timestamp, None)

        clip = frame[:4]
        self.assertListEqual(list(clip.shape), [4])
        self.assertEqual(clip.sample_rate, 11025)
        self.assertEqual(clip.timestamp, 1)


if __name__ == '__main__':
    unittest.main()
