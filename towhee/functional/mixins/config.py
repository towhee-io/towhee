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

from towhee.hparam import param_scope


class ConfigMixin:
    """
    Mixin for saving data

    Examples:

    >>> import numpy
    >>> import towhee
    >>> import time
    >>> from towhee import register, ops
    >>> @register(name='inner_distance')
    >>> def inner_distance(query, data):
    >>>     dists = []
    >>>     for vec in data:
    >>>         dist = 0
    >>>         for i in range(len(vec)):
    >>>             dist += vec[i] * query[i]
    >>>         dists.append(dist)
    >>>     return dists

    >>> t1 = time.time()
    >>> dc1 = (
    ...     towhee.dc['a']([numpy.random.random((10000, 128)) for _ in range(10)])
    ...     .set_config(enable_jit=False)
    ...     .runas_op['a', 'b'](func=lambda x: numpy.random.random(128))
    ...     .inner_distance[('b', 'a'), 'c']()
    ... )
    >>> t2 = time.time()
    >>> dc1.get_config()
    {'enable_jit': False}

    >>> t3 = time.time()
    >>> dc2 = (
    ...     towhee.dc['a']([numpy.random.random((10000, 128)) for _ in range(10)])
    ...     .enable_jit()
    ...     .runas_op['a', 'b'](func=lambda x: numpy.random.random(128))
    ...     .inner_distance[('b', 'a'), 'c']()
    ... )
    >>> t4 = time.time()
    >>> dc2.get_config()
    {'enable_jit': True}
    >>> t4-t3 < t2-t1
    True

    """
    def __init__(self) -> None:
        super().__init__()
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        if parent is not None and hasattr(parent, '_enable_jit'):
            self._enable_jit = parent._enable_jit

    def set_config(self, enable_jit: bool):
        """
        Set the parameters in DC.

        Args:
            enable_jit (`bool`):
               When set to True, this mode will speed up the Operator's function, but it may also need to return to python mode due to JIT failure,
               which will take longer, so please set it carefully.
        """
        with param_scope() as hp:
            hp().towhee.enable_jit = enable_jit
            self._enable_jit = hp.towhee.enable_jit
        return self

    def get_config(self):
        """
        Return the config in DC, such as `enable_jit`.
        """
        configs = {'enable_jit': self._enable_jit}
        return configs

    def enable_jit(self):
        """
        Set JIT to True, this mode will speed up the Operator's function, but it may also need to return to python mode due to JIT failure,
        which will take longer, so please set it carefully.
        """
        with param_scope() as hp:
            self._enable_jit = hp().towhee.enable_jit(True)
        return self

