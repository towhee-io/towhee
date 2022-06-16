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

from collections import namedtuple
from typing import Any, Dict, List

from towhee.hparam import param_scope
from towhee.engine.uri import URI


def _get_default_namespace():
    with param_scope() as hp:
        return hp().towhee.default_namespace('anon')


class OperatorRegistry:
    """Operator Registry
    """

    REGISTRY: Dict[str, Any] = {}

    def __init__(self) -> None:
        pass

    @staticmethod
    def resolve(name: str) -> Any:
        """
        Resolve operator by name
        """
        for n in [
                name,
                '{}/{}'.format(_get_default_namespace(), name),
                '{}/{}'.format('builtin', name),
        ]:
            if n in OperatorRegistry.REGISTRY:
                return OperatorRegistry.REGISTRY[n]
        return None

    @staticmethod
    def register(
            name: str = None,
            input_schema=None,  # TODO: parse input_schema from code @jie.hou
            output_schema=None,
            flag=None):
        """
        Register a class, function, or callable as a towhee operator.

        Examples:

        1. register a function as operator

        >>> from towhee import register
        >>> @register
        ... def foo(x, y):
        ...     return x+y

        2. register a class as operator

        >>> @register
        ... class foo_cls():
        ...     def __init__(self, x):
        ...         self.x = x
        ...     def __call__(self, y):
        ...         return self.x + y

        By default, function/class name is used as operator name,
        which is used by the operator factory `towhee.ops` to invoke the operator.

        >>> from towhee import ops
        >>> op = ops.foo()
        >>> op(1, 2)
        3

        >>> op = ops.foo_cls(x=2)
        >>> op(3)
        5

        3. register operator with an alternative name:

        >>> @register(name='my_foo')
        ... def foo(x, y):
        ...     return x+y
        >>> ops.my_foo()(1,2)
        3

        Operator URI and Namespace: The URI (unique reference identifier) of an operator has two parts: namespace and name.
        The namespace helps identify one operator and group the operators into various kinds.
        We can specific the namespace when create an operator:

        >>> ops.anon.my_foo()(1,2)
        3

        `anon` is the default namespace to which an operator is registered if no namespace is specified.
        And it's also the default searching namespace for the operator factory.

        You can also specific the fullname, including namespace when register an operator:

        >>> @register(name='my_namespace/my_foo')
        ... def foo(x, y):
        ...     return x+y
        >>> ops.my_namespace.my_foo()(1,2)
        3

        Output Schema:

        >>> @register(name='my_foo', output_schema='value')
        ... def foo(x, y):
        ...     return x+y
        >>> from towhee.hparam import param_scope
        >>> with param_scope('towhee.need_schema=1'):
        ...     ops.my_foo()(1,2)
        Output(value=3)

        Flag: Each operator type, for example: NNOperator and PyOperator, has their own default `flag`:

        >>> from towhee.operator.base import Operator, NNOperator, PyOperator
        >>> from towhee.operator.base import OperatorFlag
        >>> @register
        ... class foo(NNOperator):
        ...     pass
        >>> foo().flag
        <OperatorFlag.REUSEABLE|STATELESS: 6>

        The default flag can be override by `register(flag=someflag)`:

        >>> @register(flag=OperatorFlag.EMPTYFLAG)
        ... class foo(NNOperator):
        ...     pass
        >>> foo().flag
        <OperatorFlag.EMPTYFLAG: 1>

        Args:
            name (str, optional): operator name, will use the class/function name if None.
            input_schema(NamedTuple, optional): input schema for the operator. Defaults to None.
            output_schema(NamedTuple, optional): output schema, will convert the operator output to NamedTuple if not None.
            flag ([OperatorFlag], optional): operator flag. Defaults to OperatorFlag.EMPTYFLAG.

        Returns:
            [type]: [description]
        """
        if callable(name):
            # the decorator is called directly without any arguments,
            # relaunch the register
            cls = name
            return OperatorRegistry.register()(cls)

        if output_schema is None:  # none output schema
            output_schema = namedtuple('Output', 'col0')
        if isinstance(output_schema, str):  # string schema 'col0 col1'
            output_schema = output_schema.split()
        if isinstance(output_schema, List) and isinstance(output_schema[0], str):  # list schema ['col0', 'col1']
            output_schema = namedtuple('Output', output_schema)

        def wrapper(cls):
            metainfo = dict(input_schema=input_schema,
                            output_schema=output_schema,
                            flag=flag)

            nonlocal name
            name = URI(cls.__name__ if name is None else name).resolve_repo(
                _get_default_namespace())

            # wrap a callable to a class
            if not isinstance(cls, type) and callable(cls):
                func = cls
                cls = type(
                    cls.__name__, (object, ), {
                        '__call__': lambda _, *arg, **kws: func(*arg, **kws),
                        '__doc__': func.__doc__,
                    })

            if output_schema is not None:
                old_call = cls.__call__

                def wrapper_call(self, *args, **kws):
                    with param_scope() as hp:
                        need_schema = hp().towhee.need_schema(False)
                    if need_schema:
                        return output_schema(old_call(self, *args, **kws))
                    else:
                        return old_call(self, *args, **kws)

                cls.__call__ = wrapper_call
                cls.__abstractmethods__ = set()
            cls.metainfo = metainfo
            if flag is not None:
                cls.flag = property(lambda _: flag)
            OperatorRegistry.REGISTRY[name] = cls

            return cls

        return wrapper


if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=False)
