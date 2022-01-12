# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team and 2021 Zilliz.
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
import cv2

from pathlib import Path
from towhee.data.random_erase import RandomErasing

cache_path = Path(__file__).parent.parent.resolve()
test_image = cache_path.joinpath('data/dataset/kaggle_dataset_small/train/001cdf01b096e06d78e9e5112d419397.jpg')


class RandomEraseTest(unittest.TestCase):
    def test_random_erase(self):
        img = cv2.imread(str(test_image))
        RE = RandomErasing(p=1)  # pylint: disable=invalid-name
        # for i in range(20):
        #     img1 = RE(img.copy())
        #     cv2.imshow("test", img1)
        #     cv2.waitKey(1000)
        img1 = RE(img.copy())
        self.assertEqual(img.shape, img1.shape)


if __name__ == '__main__':
    unittest.main(verbosity=1)
