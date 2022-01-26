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

import importlib
from pathlib import Path
from typing import Any, Dict

from towhee.operator import Operator
from towhee.operator.nop import NOPOperator
from towhee.operator.concat_operator import ConcatOperator
from towhee.engine import LOCAL_OPERATOR_CACHE

from .operator_registry import OperatorRegistry


class OperatorLoader:
    """Wrapper class used to load operators from either local cache or a remote
    location.

    Args:
        cache_path: (`str`)
            Local cache path to use. If not specified, it will default to
            `$HOME/.towhee/operators`.
    """
    def __init__(self, cache_path: str = None):
        if cache_path is None:
            self._cache_path = LOCAL_OPERATOR_CACHE
        else:
            self._cache_path = Path(cache_path)

    def _load_interal_op(self, op_name: str, args: Dict[str, any]):
        if op_name in ['_start_op', '_end_op']:
            return NOPOperator()
        elif op_name == '_concat':
            return ConcatOperator(**args)
        else:
            # Not a interal operator
            return None

    def load_operator(self, function: str, args: Dict[str, Any], tag: str) -> Operator:
        """Attempts to load an operator from cache. If it does not exist, looks up the
        operator in a remote location and downloads it to cache instead. By standard
        convention, the operator must be called `Operator` and all associated data must
        be contained within a single directory.

        Args:
            function: (`str`)
                Origin and method/class name of the operator. Used to look up the proper
                operator in cache.
        Raises:
            FileExistsError
                Cannot find operator.
        """

        op = self._load_interal_op(function, args)
        if op is not None:
            return op

        op = OperatorRegistry.resolve(function)
        if op is not None:
            return self.instance_operator(op, args)

        module, fname = function.split('/')
        op_cls = ''.join(x.capitalize() or '_' for x in fname.split('_'))

        module = '.'.join([module, fname, fname])
        op = getattr(importlib.import_module(module), op_cls)

        return self.instance_operator(op, args)

    def instance_operator(self, op, args) -> Operator:
        return op(**args) if args is not None else op()
