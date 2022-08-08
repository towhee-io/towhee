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

from towhee.utils.log import engine_log
from towhee.hparam import param_scope
from towhee.engine.factory import ops, op
from towhee.engine.operator_registry import OperatorRegistry


class NumbaCompiler:
    """
    The just-in-time Compiler with numba.
    """

    def __init__(self, name, index, *arg, **kws):
        from towhee.utils.numba_utils import njit  # pylint: disable=import-outside-toplevel
        name_func = [name + '_func', name.replace('_', '-') + '_func']
        for n in name_func:
            func = OperatorRegistry.resolve(n)
            if func is not None:
                break
        if func is None:
            engine_log.warning(
                'The Operator: %s is not of types.FunctionType.', name)
            raise RuntimeError(
                f'The Operator: {name} is not of types.FunctionType.')
        self._func = njit(func, nogil=True)
        self._op = getattr(ops, name)[index](*arg, **kws)
        self._name = name
        self._index = index
        self._first = True
        self._success = True

    def __apply__(self, *arg):
        # Multi inputs.
        if isinstance(self._index[0], tuple):
            args = [getattr(arg[0], x) for x in self._index[0]]
        # Single input.
        else:
            args = [getattr(arg[0], self._index[0])]
        return self._func(*args)

    def jit_call(self, *arg, **kws):
        if bool(kws):
            engine_log.warning(
                'The compiled function in Numba does not support kwargs.')
            raise KeyError(
                'The compiled function in Numba does not support kwargs.')
        if bool(self._index):
            res = self.__apply__(*arg)

            # Multi outputs.
            if isinstance(res, tuple):
                if not isinstance(self._index[1],
                                  tuple) or len(self._index[1]) != len(res):
                    raise IndexError(
                        f'Op has {len(res)} outputs, but {len(self._index[1])} indices are given.'
                    )
                for i, j in zip(self._index[1], res):
                    setattr(arg[0], i, j)
            # Single output.
            else:
                setattr(arg[0], self._index[1], res)

            return arg[0]
        else:
            res = self._func(*arg)
            return res

    def __call__(self, *arg, **kws):
        if self._first:
            self._first = False
            try:
                return self.jit_call(*arg, **kws)
            except Exception as e:  # pylint: disable=broad-except
                self._success = False
                engine_log.warning(
                    'Failed to speed up your function:%s with error:%s in JIT mode, will back to Python interpreter.',
                    self._name, e)
                return self._op.__call__(*arg, **kws)
        elif self._success:
            return self.jit_call(*arg, **kws)
        else:
            return self._op.__call__(*arg, **kws)


class TowheeCompiler:
    """
    Towhee's just-in-time compiler
    """

    def __init__(self, name, index, *arg, **kws):
        from towhee.compiler import jit_compile  # pylint: disable=import-outside-toplevel
        # self._op = getattr(ops, name)[index](*arg, **kws)
        self._name = name
        self._index = index
        self._compiler = jit_compile(feature=True)
        op_name = self._name.replace('.', '/').replace('_', '-')
        with param_scope(index=self._index):
            self._op = op(op_name, 'main', arg, kws)

    def set_compiler(self, compiler):
        self._compiler = compiler

    def __apply__(self, *arg):
        # Multi inputs.
        if isinstance(self._index[0], tuple):
            args = [getattr(arg[0], x) for x in self._index[0]]
        # Single input.
        else:
            args = [getattr(arg[0], self._index[0])]
        with self._compiler:
            return self._op(*args)

    def __call__(self, *arg, **kws):
        if bool(self._index):
            res = self.__apply__(*arg)

            # Multi outputs.
            if isinstance(res, tuple):
                if not isinstance(self._index[1],
                                  tuple) or len(self._index[1]) != len(res):
                    raise IndexError(
                        f'Op has {len(res)} outputs, but {len(self._index[1])} indices are given.'
                    )
                for i, j in zip(self._index[1], res):
                    setattr(arg[0], i, j)
            # Single output.
            else:
                setattr(arg[0], self._index[1], res)

            return arg[0]
        else:
            with self._compiler:
                return self._op.__call__(*arg, **kws)


class CompileMixin:
    """
    Mixin to just-in-time complie the Operator.

    Examples:

    >>> import numpy
    >>> import towhee
    >>> import time
    >>> from towhee import register
    >>> @register(name='inner_distance')
    ... def inner_distance(query, data):
    ...     dists = []
    ...     for vec in data:
    ...         dist = 0
    ...         for i in range(len(vec)):
    ...             dist += vec[i] * query[i]
    ...         dists.append(dist)
    ...     return dists
    >>> data = [numpy.random.random((10000, 128)) for _ in range(10)]
    >>> query = numpy.random.random(128)

    >>> t1 = time.time()
    >>> dc1 = (
    ...     towhee.dc['a'](data)
    ...     .runas_op['a', 'b'](func=lambda _: query)
    ...     .inner_distance[('b', 'a'), 'c']()
    ... )
    >>> t2 = time.time()
    >>> dc2 = (
    ...     towhee.dc['a'](data)
    ...     .config(jit='numba')
    ...     .runas_op['a', 'b'](func=lambda _: query)
    ...     .inner_distance[('b', 'a'), 'c']()
    ... )
    >>> t3 = time.time()
    >>> assert(t3-t2 < t2-t1)
    """

    def __init__(self) -> None:
        super().__init__()
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        if parent is not None and hasattr(parent, '_jit'):
            self._jit = parent._jit

    def set_jit(self, compiler, **kws):  # pylint: disable=unused-argument
        if compiler in ['numba', 'towhee']:
            self._jit = compiler
        else:
            engine_log.error(
                'Error when setting jit, please make sure the configuration about jit in [\'numba\'].'
            )
            raise KeyError(
                'Error when setting jit, please make sure the configuration about jit in [\'numba\'].'
            )
        return self

    def jit_resolve(self, name, index, *arg, **kws):
        try:
            if isinstance(self._jit, str):
                if self._jit == 'numba':
                    return NumbaCompiler(name, index, *arg, **kws)
                if self._jit == 'towhee':
                    return TowheeCompiler(name, index, *arg, **kws)
            else:
                retval = TowheeCompiler(name, index, *arg, **kws)
                retval.set_compiler(self._jit)
                return retval
        except:  # pylint: disable=bare-except
            return self.resolve(name, index, *arg, **kws)
