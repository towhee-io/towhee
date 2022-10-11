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

import towhee.datacollection.data_collection


def load_tests(loader, tests, ignore):
    # pylint: disable=unused-argument
    tests.addTests(doctest.DocTestSuite(towhee.datacollection.data_collection))
    return tests


if __name__ == '__main__':
    unittest.main()
