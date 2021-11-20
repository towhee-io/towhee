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


import importlib.util
from pathlib import Path
from typing import Any, Dict

from towhee.operator import Operator
from towhee.operator.nop import NOPOperator
from towhee.engine import LOCAL_OPERATOR_CACHE
from towhee.hub.file_manager import FileManager


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

    def load_operator(self, function: str, args: Dict[str, Any]) -> Operator:
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

        if function in ['_start_op', '_end_op']:
            return NOPOperator()

        fm = FileManager()
        path = fm.get_operator(function)

        if path is None:
            raise FileExistsError('Cannot find operator.')

        fname = str(path).rsplit('/', maxsplit=1)[-1][:-3]
        modname = 'towhee.operator.' + fname

        spec = importlib.util.spec_from_file_location(
            modname, path.resolve())

        # Create the module and then execute the module in its own namespace.
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Instantiate the operator object and return it to the caller for
        # `load_operator`. By convention, the operator class is simply the CamelCase
        # version of the snake_case operator.
        op_cls = ''.join(x.capitalize() or '_' for x in fname.split('_'))
        if args is not None:
            return getattr(module, op_cls)(**args)
        else:
            return getattr(module, op_cls)()
