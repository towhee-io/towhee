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

import random
import math

import numpy


class RandomErasing:
    """
    Random erasing the an rectangle region in Image.
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.

    Args:
        sl (`float`): min erasing area region
        sh (`float`): max erasing area region
        r1 (`float`): min aspect ratio range of earsing region
        p (`float`): probability of performing random erasing
    """

    def __init__(self, p: float = 0.5, sl: float = 0.02, sh: float = 0.4, r1: float = 0.3):
        self.p = p
        self.s = (sl, sh)
        self.r = (r1, 1 / r1)

    def __call__(self, img: numpy.ndarray):
        """
        perform random erasing
        Args:
            img (`numpy.ndarray`): opencv numpy array in form of [w, h, c] range
                 from [0, 255]

        Returns:
            (`numpy.ndarray`)
                erased img.
        """

        assert len(img.shape) == 3, 'image should be a 3 dimension numpy array'

        if random.random() > self.p:
            return img

        else:
            while True:
                se = random.uniform(*self.s) * img.shape[0] * img.shape[1]
                re = random.uniform(*self.r)

                he = int(round(math.sqrt(se * re)))
                we = int(round(math.sqrt(se / re)))

                xe = random.randint(0, img.shape[1])
                ye = random.randint(0, img.shape[0])

                if xe + we <= img.shape[1] and ye + he <= img.shape[0]:
                    img[ye: ye + he, xe: xe + we, :] = numpy.random.randint(low=0, high=255, size=(he, we, img.shape[2]))

                    return img
