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

from typing import Union
from towhee.hparam import param_scope


class ConfigMixin:
    """
    Mixin to config DC, such as set the `parallel`, `chunksize`, `jit`.

    Examples:

    >>> import towhee
    >>> dc = towhee.dc['a'](range(20))
    >>> dc = dc.set_chunksize(10)
    >>> dc = dc.set_parallel(2)
    >>> dc = dc.set_jit('numba')
    >>> dc.get_config()
    {'parallel': 2, 'chunksize': 10, 'jit': 'numba'}
    >>> dc1 = towhee.dc([1,2,3]).config(jit='numba')
    >>> dc2 = towhee.dc['a'](range(40)).config(parallel=2, chunksize=20)
    >>> dc1.get_config()
    {'parallel': None, 'chunksize': None, 'jit': 'numba'}
    >>> dc2.get_config()
    {'parallel': 2, 'chunksize': 20, 'jit': None}
    """

    def __init__(self) -> None:
        super().__init__()
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        if parent is not None and hasattr(parent, '_config'):
            self._config = parent._config
        else:
            self._config = None
        if parent is None or not hasattr(parent, '_num_worker'):
            self._num_worker = None
        if parent is None or not hasattr(parent, '_chunksize'):
            self._chunksize = None
        if parent is None or not hasattr(parent, '_jit'):
            self._jit = None

    def config(self, parallel: int = None, chunksize: int = None, jit: Union[str, dict] = None):
        """
        Set the parameters in DC.

        Args:
            parallel (`int`):
               Set the number of parallel execution for following calls.
            chunksize (`int`):
               Set the chunk size for arrow.
            jit (`Union[str, dict]`):
               It can set to "numba", this mode will speed up the Operator's function, but it may also need to return to python mode due to JIT
               failure, which will take longer, so please set it carefully.
        """
        dc = self
        if jit is not None:
            dc = dc.set_jit(compiler=jit)
        if parallel is not None:
            dc = dc.set_parallel(num_worker=parallel)
        if chunksize is not None:
            dc = dc.set_chunksize(chunksize=chunksize)
        return dc

    def get_config(self):
        """
        Return the config in DC, such as `parallel`, `chunksize` and `jit`.
        """
        self._config = {
            'parallel': self._num_worker,
            'chunksize': self._chunksize,
            'jit': self._jit,
        }
        return self._config
