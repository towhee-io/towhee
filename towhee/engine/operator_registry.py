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


class OperatorRegistry:
    """Operator Registry
    """

    REGISTRY: Dict[str, Any] = {}

    def __init__(self) -> None:
        pass

    @staticmethod
    def register(name: str, op: Any) -> None:
        OperatorRegistry.REGISTRY[name] = op

    @staticmethod
    def resolve(name: str) -> Any:
        if name in OperatorRegistry.REGISTRY:
            return OperatorRegistry.REGISTRY[name]
        return None

    @staticmethod
    def operator(version,
                 name: str = None,
                 author: str = None,
                 description: str = None,
                 long_description: str = None,
                 imports: List[str] = [],
                 input_schema=None,
                 output_schema=None):
        metainfo = {}
        def wrapper(cls):
            cls.metainfo = metainfo
            OperatorRegistry.register(name, cls)
            return cls

        return wrapper