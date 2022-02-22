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
import concurrent.futures
from queue import Queue


class ParallelMixin:
    """
    Mixin for parallel execution.

    Examples:
    >>> from towhee.functional import DataCollection
    >>> def add_1(x):
    ...     x+1
    >>> result = DataCollection.range(1000).map(add_1).parallel(5).to_list()
    >>> len(result)
    1000
    """

    def parallel(self, num_worker):
        executor = concurrent.futures.ThreadPoolExecutor(num_worker)
        queue = Queue(maxsize=num_worker)
        gen = iter(self)
        cnt = num_worker

        def worker():
            nonlocal cnt
            for x in gen:
                queue.put(x)
            cnt -= 1

        for _ in range(num_worker):
            executor.submit(worker)

        def inner():
            while cnt > 0 or not queue.empty():
                yield queue.get()
            executor.shutdown()

        return self.factory(inner())


if __name__ == '__main__':  # pylint: disable=inconsistent-quotes
    import doctest

    doctest.testmod(verbose=False)
