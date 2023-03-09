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

import types
from typing import Any, Dict

from towhee.operator import SharedType
from .uri import URI


class OperatorRegistry:
    """
    Operator Registry
    """

    REGISTRY: Dict[str, Any] = {}

    @staticmethod
    def resolve(name: str) -> Any:
        """
        Resolve operator by name
        """
        for n in [
                name,
                '{}/{}'.format('anon', name),
                '{}/{}'.format('builtin', name),
        ]:
            if n in OperatorRegistry.REGISTRY:
                return OperatorRegistry.REGISTRY[n]
        return None

    @staticmethod
    def register(
            name: str = None,
            input_schema=None,  # for legacy op
            output_schema=None, # for legacy op
            flag=None # for legacy op
    ):
        """
        Register a class, function, or callable as a towhee operator.

        Args:
            name (str, optional): operator name, will use the class/function name if None.

        Returns:
            [type]: [description]
        """
        if callable(name):
            # the decorator is called directly without any arguments,
            # relaunch the register
            cls = name
            return OperatorRegistry.register()(cls)

        def wrapper(cls):
            nonlocal name
            name = URI(cls.__name__ if name is None else name).resolve_repo('anon')

            if isinstance(cls, types.FunctionType):
                OperatorRegistry.REGISTRY[name + '_func'] = cls

            # wrap a callable to a class
            if not isinstance(cls, type) and callable(cls):
                func = cls
                cls = type(
                    cls.__name__, (object, ), {
                        '__call__': lambda _, *arg, **kws: func(*arg, **kws),
                        '__doc__': func.__doc__,
                    })

            if not hasattr(cls, 'shared_type'):
                cls.shared_type = SharedType.Shareable
            OperatorRegistry.REGISTRY[name] = cls

            return cls

        return wrapper

    @staticmethod
    def op_names():
        return list(OperatorRegistry.REGISTRY.keys())
