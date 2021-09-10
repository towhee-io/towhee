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


class OperatorLoader:
    """Wrapper class used to load operators from either local cache or a remote
    location.

    Args:
        cache_path: (`str`)
            Local cache path to use. If not specified, it will default to
            `$HOME/.towhee/cache`.
    """

    def __init__(self, cache_path: str = None):
        if not cache_path:
            self._cache_path = Path.home() / '.towhee/cache'
        else:
            self._cache_path = Path(cache_path)
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
        """

        # Lookup the path for the operator in local cache. Example directory structure:
        #  /home/user/.towhee/cache/organization-name/operator-name
        #   |_ /home/user/.towhee/cache/organization-name/operator-name/operator.py
        #   |_ /home/user/.towhee/cache/organization-name/operator-name/resnet.torch
        #   |_ /home/user/.towhee/cache/organization-name/operator-name/config.json
        fname = function.split('/')[-1]
        path = self._cache_path / function / (fname + '.py')

        # If the following check passes, the desired operator was found locally and can
        # be loaded from cache.
        if path.is_file():

            # Specify the module name and absolute path of the file that needs to be
            # imported.
            modname = 'towhee.operator.' + fname
            spec = importlib.util.spec_from_file_location(modname, path.resolve())

            # Create the module and then execute the module in its own namespace.
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Instantiate the operator object and return it to the caller for
            # `load_operator`. By convention, the operator class is simply the CamelCase
            # version of the snake_case operator.
            op_cls = ''.join(x.capitalize() or '_' for x in fname.split('_'))
            return getattr(module, op_cls)(**args)

        else:
            raise FileNotFoundError('Operator definition not found')
