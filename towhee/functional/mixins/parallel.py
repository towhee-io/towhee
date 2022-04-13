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
import asyncio
import threading

from towhee.utils.log import engine_log
from towhee.functional.option import Option, Empty, _Reason
from towhee.hparam.hyperparameter import param_scope


def _map_task(x, unary_op):
    def map_wrapper():
        try:
            if isinstance(x, Option):
                return x.map(unary_op)
            else:
                return unary_op(x)
        except Exception as e:  # pylint: disable=broad-except
            engine_log.warning(f'{e}, please check {x} with op {unary_op}. Continue...')  # pylint: disable=logging-fstring-interpolation
            return Empty(_Reason(x, e))

    return map_wrapper


class ParallelMixin:
    """
    Mixin for parallel execution.

    Examples:

    >>> from towhee import DataCollection
    >>> def add_1(x):
    ...     return x+1
    >>> result = DataCollection.range(1000).set_parallel(2).map(add_1).to_list()
    >>> len(result)
    1000

    >>> result = DataCollection.range(1000).pmap(add_1, 10).pmap(add_1, 10).to_list()
    >>> result[990:]
    [992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001]
    """

    def __init__(self) -> None:
        super().__init__()
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        if parent is not None and hasattr(parent, '_executor') and isinstance(parent._executor, concurrent.futures.ThreadPoolExecutor):
            self._backend = parent._backend
            self._executor = parent._executor
            self._num_worker = parent._num_worker

    def get_executor(self):
        if hasattr(self, '_executor') and isinstance(self._executor, concurrent.futures.ThreadPoolExecutor) :
            return self._executor
        return None

    def get_backend(self):
        if hasattr(self, '_backend')  and isinstance(self._backend, str):
            return self._backend
        return None

    def get_num_worker(self):
        if hasattr(self, '_num_worker')  and isinstance(self._num_worker, int):
            return self._num_worker
        return None

    def set_parallel(self, num_worker = 2, backend = 'thread'):
        """
        Set parallel execution for following calls.

        Examples:

        >>> from towhee import DataCollection
        >>> import threading
        >>> stage_1_thread_set = set()
        >>> stage_2_thread_set = set()
        >>> result = (
        ...     DataCollection.range(1000).stream().set_parallel(4)
        ...     .map(lambda x: stage_1_thread_set.add(threading.current_thread().ident))
        ...     .map(lambda x: stage_2_thread_set.add(threading.current_thread().ident)).to_list()
        ... )

        >>> len(stage_2_thread_set)>1
        True
        """
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_worker)
        self._backend = backend
        self._num_worker = num_worker
        return self

    def unset_parallel(self):
        """
        Unset parallel execution for following calls.

        Examples:

        >>> from towhee import DataCollection
        >>> import threading
        >>> stage_1_thread_set = {threading.current_thread().ident}
        >>> stage_2_thread_set = {threading.current_thread().ident}
        >>> result = (
        ...     DataCollection.range(1000).stream().set_parallel(4)
        ...     .map(lambda x: stage_1_thread_set.add(threading.current_thread().ident))
        ...     .unset_parallel()
        ...     .map(lambda x: stage_2_thread_set.add(threading.current_thread().ident)).to_list()
        ... )

        >>> len(stage_1_thread_set)>1
        True
        >>> len(stage_2_thread_set)>1
        False
        """
        self._backend = None
        self._num_wokrker = None
        self._executor = None
        return self

    def split(self, count):
        """
        Split a dataframe into multiple dataframes.

        Args:
            count (int): how many resulting DCs;

        Returns:
            [DataCollection, ...]: copies of DC;

        Examples:

        1. Split:
        >>> from towhee import DataCollection
        >>> dc = DataCollection([0, 1, 2, 3, 4]).stream()
        >>> a, b, c = dc.split(3)
        >>> a.zip(b, c).to_list()
        [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)]
        """
        if self.is_stream:
            queues = [Queue(maxsize=1) for _ in range(count)]
        else:
            queues = [Queue() for _ in range(count)]
        loop = asyncio.new_event_loop()

        def inner(queue):
            while True:
                x = queue.get()
                if isinstance(x, EOS):
                    break
                else:
                    yield x

        async def worker():
            for x in self:
                for queue in queues:
                    queue.put(x)
            for queue in queues:
                poison = EOS()
                queue.put(poison)

        def worker_wrapper():
            loop.run_until_complete(worker())
            loop.close()

        t = threading.Thread(target=worker_wrapper)
        t.start()
        retval = [inner(queue) for queue in queues]
        return [self._factory(x) for x in retval]

    def pmap(self, unary_op, num_worker = None, backend = None):
        """
        Apply `unary_op` with parallel execution.
        Currently supports two backends, `ray` and `thread`.

        Args:
            unary_op (func): the op to be mapped;
            num_worker (int): how many threads to reserve for this op;
            backend (str): whether to use `ray` or `thread`

        Examples:

        >>> from towhee import DataCollection
        >>> import threading
        >>> stage_1_thread_set = {threading.current_thread().ident}
        >>> stage_2_thread_set = {threading.current_thread().ident}
        >>> result = (
        ...     DataCollection.range(1000).stream()
        ...     .pmap(lambda x: stage_1_thread_set.add(threading.current_thread().ident), 5)
        ...     .pmap(lambda x: stage_2_thread_set.add(threading.current_thread().ident), 4).to_list()
        ... )
        >>> len(stage_1_thread_set) > 1
        True
        >>> len(stage_2_thread_set) > 1
        True
        """
        if backend is None:
            if self.get_backend() == 'ray':
                return self._ray_pmap(unary_op, num_worker)
            else:
                return self._thread_pmap(unary_op, num_worker)
        elif backend == 'thread':
            return self._thread_pmap(unary_op, num_worker)
        elif backend == 'ray':
            return self._ray_pmap(unary_op, num_worker)

    def _thread_pmap(self, unary_op, num_worker=None):
        if num_worker is not None:
            executor = concurrent.futures.ThreadPoolExecutor(num_worker)
        elif self.get_executor() is not None:
            executor = self._executor
            num_worker = self._num_worker
        else:
            executor = concurrent.futures.ThreadPoolExecutor(2)
            num_worker = 2

        #If not streamed, we need to be able to hold all values within queue
        if self.is_stream:
            queue = Queue(num_worker)
        else:
            queue = Queue()

        loop = asyncio.new_event_loop()

        def inner():
            while True:
                x = queue.get()
                if isinstance(x, EOS):
                    break
                else:
                    yield x

        async def worker():
            buff = []
            for x in self:
                if len(buff) == num_worker:
                    queue.put(await buff.pop(0))
                buff.append(loop.run_in_executor(executor, _map_task(x, unary_op)))
            while len(buff) > 0:
                queue.put(await buff.pop(0))
            queue.put(EOS())

        def worker_wrapper():
            loop.run_until_complete(worker())
            loop.close()

        t = threading.Thread(target=worker_wrapper)
        t.start()

        return self._factory(inner())

    def mmap(self, ops: list, num_worker = None, backend = None):
        """
        Apply multiple unary_op to data collection.
        Currently supports two backends, `ray` and `thread`.

        Args:
            unary_op (func): the op to be mapped;
            num_worker (int): how many threads to reserve for this op;
            backend (str): whether to use `ray` or `thread`

        Examples:

        1. Using mmap:

        >>> from towhee import DataCollection
        >>> dc = DataCollection([0,1,2,'3',4]).stream()
        >>> a, b = dc.mmap([lambda x: x+1, lambda x: x*2])
        >>> c = a.map(lambda x: x+1)
        >>> c.zip(b).to_list()
        [(2, 0), (3, 2), (4, 4), (Empty(), '33'), (6, 8)]

        2. Using map instead of mmap:

        >>> from towhee import DataCollection
        >>> dc = DataCollection.range(5).stream()
        >>> a, b, c = dc.map(lambda x: x+1, lambda x: x*2, lambda x: int(x/2))
        >>> d = a.map(lambda x: x+1)
        >>> d.zip(b, c).to_list()
        [(2, 0, 0), (3, 2, 0), (4, 4, 1), (5, 6, 1), (6, 8, 2)]

        3. DAG execution:

        >>> dc = DataCollection.range(5).stream()
        >>> a, b, c = dc.map(lambda x: x+1, lambda x: x*2, lambda x: int(x/2))
        >>> d = a.map(lambda x: x+1)
        >>> d.zip(b, c).map(lambda x: x[0]+x[1]+x[2]).to_list()
        [2, 5, 9, 12, 16]
        """
        if len(ops) == 1:
            return self._pmap(unary_op=ops[0], num_worker=num_worker, backend=backend)

        next_vals = []
        next_vals = self.split(len(ops))

        ret = []
        for i, x in enumerate(ops):
            ret.append(next_vals[i].pmap(x, num_worker=num_worker, backend=backend))
        return ret

class EOS():
    '''
    Internal object used to signify end of processing queue.
    '''
    pass


if __name__ == '__main__':  # pylint: disable=inconsistent-quotes
    import doctest
    doctest.testmod(verbose=False)
