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


import numpy as np
from towhee.array.array import Array


def full(shape: int, fill_value):
    """Return a new `Array` of given shape filled with `fill_value`

    Args:
        shape: (`int`)
            Shape of the new `Array`
        fill_value: (`numpy.dtype` or `str`)
            Fill value
    """
    return Array(data=np.full(shape=shape, fill_value=fill_value))
