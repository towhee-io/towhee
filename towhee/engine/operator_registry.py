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
        if name in OperatorRegistry.REGISTRY:
            return OperatorRegistry.REGISTRY[name]
        return None

    @staticmethod
    def register(
        version,
        name: str = None,
        author: str = 'towhee_community',
        description: str = None,
        long_description: str = None,
        imports: List[str] = None,  # TODO: parse imports from code @jie.hou
        input_schema=None,  # TODO: parse input_schema from code @jie.hou
        output_schema=None,
        shared_type=SharedType.Shareable,
    ):
        """register a class as towhee operator

        Args:
            version (str): operator version
            name (str, optional): operator name, will use the class name if None.
            author (str, optional): operator author. Defaults to 'towhee_community'.
            description (str, optional): description of the operator, use the first line of the class's doc string if  None.
            long_description (str, optional): long description to the operator, use class doc string if None.
            imports (List[str], optional): python module dependency. Defaults to None.
            input_schema(NamedTuple, optional): input schema for the operator. Defaults to None.
            output_schema(NamedTuple, optional): output schema, will convert the operator output to NamedTuple if not None.
            shared_type ([type], optional): operator shared_type. Defaults to SharedType.Shareable.

        Returns:
            [type]: [description]
        """
        # TODO: need to convert the class name to URI @shiyu22
        name = name.replace('_', '-')

        def wrapper(cls):
            metainfo = dict(
                version=version,
                author=author if author is not None else 'towhee_community',
                description= \
                    description if description is not None else OperatorRegistry.parse_description(cls),
                long_description= \
                    long_description if long_description is not None else cls.__doc__,
                imports = imports if imports is not None else [],
                input_schema = input_schema if input_schema is not None else input_schema,
                shared_type=shared_type)

            # wrap a callable to a class
            if not isinstance(cls, type) and callable(cls):
                old_cls = cls

                class WrapperClass:  # TODO: generate the class name from function name @jie.hou

                    def __init__(self, *arg, **kws) -> None:
                        pass

                    def __call__(self, *arg, **kws):
                        return old_cls(*arg, **kws)

                cls = WrapperClass

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
            return cls

        return wrapper

    @staticmethod
    def parse_description(op):
        return '' if op.__doc__ is None else op.__doc__.split('\n')[0]
