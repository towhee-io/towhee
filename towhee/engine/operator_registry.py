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

from towhee.operator.base import SharedType
from towhee.hparam import param_scope


class OperatorRegistry:
    """Operator Registry
    """

    REGISTRY: Dict[str, Any] = {}

    def __init__(self) -> None:
        pass

    @staticmethod
    def resolve(name: str) -> Any:
        with param_scope() as hp:
            default_namespace = hp().towhee.default_namespace('anon')
        if name in OperatorRegistry.REGISTRY:
            return OperatorRegistry.REGISTRY[name]

        name_with_ns = '{}/{}'.format(default_namespace, name)
        if name_with_ns in OperatorRegistry.REGISTRY:
            return OperatorRegistry.REGISTRY[name_with_ns]

        name_with_ns = '{}/{}'.format('builtin', name)
        if name_with_ns in OperatorRegistry.REGISTRY:
            return OperatorRegistry.REGISTRY[name_with_ns]
        return None

    @staticmethod
    def register(
        name: str = None,
        input_schema=None,  # TODO: parse input_schema from code @jie.hou
        output_schema=None,
        shared_type=SharedType.Shareable,
    ):
        """register a class or callable as a towhee operator.

        1. Basic operator registration and calling
        The registration is as simple as just adding one line before a function
        >>> from towhee import register
        >>> @register
        ... def foo(x, y):
        ...     return x+y

        or register a class as a stateful operator
        >>> @register
        ... class foo_cls():
        ...     def __init__(self, x):
        ...         self.x = x
        ...     def __call__(self, y):
        ...         return self.x + y

        By default, function/class name is used as operator name.
        After registration, we are able to call the operator by its name:
        >>> from towhee import ops
        >>> op = ops.foo()
        >>> op(1, 2)
        3

        or calling a stateful operator
        >>> op = ops.foo_cls(x=2)
        >>> op(3)
        5

        2. Register with detail information
        Operator name can also be provided during registration:
        >>> @register(name='my_foo')
        ... def foo(x, y):
        ...     return x+y
        >>> ops.my_foo()(1,2)
        3

        Each operator has its namespace. If not provided, default namespace 'anon' would be used.
        You can also specific the fullname, including namespace when creating an operator.
        >>> ops.anon.my_foo()(1,2)
        3

        or the default namespace, typically `anon` will be searched by the factory method. To register
        the operator to a different namespace, you should add namespace as a prefix of `name`:
        >>> @register(name='my_namespace/my_foo')
        ... def foo(x, y):
        ...     return x+y
        >>> ops.my_namespace.my_foo()(1,2)
        3

        The operator can also has its outout schema
        >>> @register(name='my_foo', output_schema='value')
        ... def foo(x, y):
        ...     return x+y
        >>> from towhee.hparam import param_scope
        >>> with param_scope('towhee.need_schema=1'):
        ...     ops.my_foo()(1,2)
        Output(value=3)

        Args:
            name (str, optional): operator name, will use the class/function name if None.
            input_schema(NamedTuple, optional): input schema for the operator. Defaults to None.
            output_schema(NamedTuple, optional): output schema, will convert the operator output to NamedTuple if not None.
            shared_type ([type], optional): operator shared_type. Defaults to SharedType.Shareable.

        Returns:
            [type]: [description]
        """
        if callable(name):
            return OperatorRegistry.register()(name)

        if output_schema is None:  # none output schema
            output_schema = namedtuple('Output', 'col0')
        if isinstance(output_schema, str):  # string schema 'col0 col1'
            output_schema = output_schema.split()
        if isinstance(output_schema, List):  # list schema ['col0', 'col1']
            output_schema = namedtuple('Output', output_schema)

        with param_scope() as hp:
            default_namespace = hp().towhee.default_namespace('anon')

        def wrapper(cls):
            metainfo = dict(input_schema=input_schema,
                            output_schema=output_schema,
                            shared_type=shared_type)

            # TODO: need to convert the class name to URI @shiyu22
            nonlocal name
            if name is None:
                name = cls.__name__
            if name is not None:
                name = name.replace('_', '-')
            if '/' not in name:
                name = '{}/{}'.format(default_namespace, name)

            # wrap a callable to a class
            if not isinstance(cls, type) and callable(cls):
                old_cls = cls

                class WrapperClass:  # TODO: generate the class name from function name @jie.hou

                    def __init__(self, *arg, **kws) -> None:
                        pass

                    def __call__(self, *arg, **kws):
                        return old_cls(*arg, **kws)

                cls = WrapperClass

            else:
                old_cls = cls

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
            cls.shared_type = property(lambda self: shared_type)
            OperatorRegistry.REGISTRY[name] = cls
            if hasattr(old_cls, '__doc__'):  # pylint: disable=inconsistent-quotes
                cls.__doc__ = old_cls.__doc__
            return cls

        return wrapper


if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=False)
