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
import doctest
import unittest

import numpy as np

from towhee.functional import DataCollection

import towhee.functional.mixins.parallel


class TestParallel(unittest.TestCase):
    """
    tests for data collection parallel execution
    """

    def test_example_for_basic_api(self):
        _ = (
            DataCollection.range(50).stream() \
                .pmap(lambda x: np.random.random((x * 20, x * 20)),3) \
                .pmap(lambda x: np.dot(x, x), 4).to_list()
        )
        _ = (
            DataCollection.range(10).stream().safe()\
                .pmap(lambda x: np.random.random((x * 20, x * 20)), 3) \
                .pmap(lambda x: np.dot(x, x), 4).to_list()
        )


TestDataCollectionParallelExecution = doctest.DocTestSuite(
    towhee.functional.mixins.parallel)
unittest.TextTestRunner().run(TestDataCollectionParallelExecution)

if __name__ == '__main__':
    unittest.main()
