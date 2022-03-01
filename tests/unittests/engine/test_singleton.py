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
import queue
import time

from towhee.utils.singleton import singleton
from tests.unittests.test_util.dataframe_test_util import MultiThreadRunner


@singleton
class Singleton:
    def __init__(self):
        pass


def create_obj(q):
    for _ in range(10):
        q.put(Singleton())
        time.sleep(0.01)


class TestSingleton(unittest.TestCase):
    """
    Singleton decorator
    """

    def test_singleton(self):
        q = queue.Queue()
        runner = MultiThreadRunner(create_obj, (q, ), 10)
        runner.start()
        runner.join()

        self.assertEqual(q.qsize(), 100)

        while not q.empty():
            self.assertEqual(q.get(), Singleton())

if __name__ == '__main__':
    unittest.main()
