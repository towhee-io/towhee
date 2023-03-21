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

# pylint: disable=unused-import
# pylint: disable=dangerous-default-value

from typing import Any, Dict, Tuple, Callable

from towhee.runtime.operator_manager import OperatorLoader, OperatorRegistry
from towhee.utils.log import engine_log


class OpsParser:
    """
    Runtime parsing of unknown attributes.

    Example:
    >>> from towhee.runtime.factory import ops_parse
    >>> @ops_parse
    ... def foo(name, *args, **kwargs):
    ...     return str((name, args, kwargs))
    >>> print(foo.bar.zed(1, 2, 3))
    ('bar.zed', (1, 2, 3), {})
    """

    def __init__(self, func: Callable, name=None):
        self._func = func
        self._name = name

    def __call__(self, *args, **kws) -> Any:
        return self._func(self._name, *args, **kws)

    def __getattr__(self, name: str) -> Any:
        if self._name is not None:
            name = f'{self._name}.{name}'
        return ops_parse(self._func, name)


def ops_parse(func, name=None):
    """
    Wraps function with a class to allow __getattr__ on a function.
    """
    new_class = type(func.__name__, (
        OpsParser,
        object,
    ), dict(__doc__=func.__doc__))
    return new_class(func, name)


class _OperatorWrapper:
    """
    Operator wrapper for initialization.
    """

    def __init__(self,
                 name: str,
                 init_args: Tuple = None,
                 init_kws: Dict[str, Any] = None,
                 tag: str = 'main',
                 latest: bool = False):
        self._name = name.replace('.', '/').replace('_', '-')
        self._tag = tag
        self._latest = latest
        self._init_args = init_args
        self._init_kws = init_kws
        self._op = None

    @property
    def name(self):
        return self._name

    @property
    def tag(self):
        return self._tag

    @property
    def init_args(self):
        return self._init_args

    @property
    def init_kws(self):
        return self._init_kws

    @property
    def is_latest(self):
        return self._latest

    def revision(self, tag: str = 'main'):
        self._tag = tag
        return self

    def latest(self):
        self._latest = True
        return self

    def get_op(self):
        if self._op is None:
            self.preload_op()
        return self._op

    def preload_op(self):
        try:
            loader = OperatorLoader()
            self._op = loader.load_operator(self._name, self._init_args, self._init_kws, tag=self._tag, latest=self._latest)
        except Exception as e:
            err = f'Loading operator with error:{e}'
            engine_log.error(err)
            raise RuntimeError(err) from e

    def __call__(self, *args, **kws):
        if self._op is None:
            self.preload_op()

        result = self._op(*args, **kws)
        return result

    @staticmethod
    def callback(name: str, *args, **kws):
        return _OperatorWrapper(name, args, kws)


class _OperatorParser:
    """
    Class to loading operator with _OperatorWrapper.
    """
    @classmethod
    def __getattr__(cls, name):
        @ops_parse
        def wrapper(name, *args, **kws):
            return _OperatorWrapper.callback(name, *args, **kws)

        return getattr(wrapper, name)


ops = _OperatorParser()

register = OperatorRegistry.register
