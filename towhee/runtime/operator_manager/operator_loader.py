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
import sys
import subprocess
from pathlib import Path
from typing import Any, List, Dict, Union
import re
import traceback
import pkg_resources

from towhee.operator import Operator
from towhee.operator.nop import NOPNodeOperator
from towhee.hub import get_operator
from towhee.runtime.constants import InputConst, OutputConst
from towhee.utils.log import engine_log
from .operator_registry import OperatorRegistry


class OperatorLoader:
    """
    Wrapper class used to load operators from either local cache or a remote
    location.

    Args:
        cache_path: (`str`)
            Local cache path to use. If not specified, it will default to
            `$HOME/.towhee/operators`.
    """

    def _load_operator_from_internal(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        if function in [InputConst.name, OutputConst.name]:
            return NOPNodeOperator()
        else:
            return None

    def _load_operator_from_registry(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        op = OperatorRegistry.resolve(function)
        return self._instance_operator(op, arg, kws) if op is not None else None

    def _load_legacy_op(self, modname, path, fname):
        # support old version operator API
        file_name = path / (fname + '.py')
        spec = importlib.util.spec_from_file_location(modname, file_name.resolve())
        # Create the module and then execute the module in its own namespace.

        module = importlib.util.module_from_spec(spec)
        sys.modules[modname] = module
        spec.loader.exec_module(module)
        # Instantiate the operator object and return it to the caller for
        # `load_operator`. By convention, the operator class is simply the CamelCase
        # version of the snake_case operator.
        op_cls = ''.join(x.capitalize() or '_' for x in fname.split('_'))
        op = getattr(module, op_cls)
        return op

    def _load_op(self, modname, path, fname):
        # support latest version operator API
        init_path = path / '__init__.py'
        if not init_path.is_file():
            return None
        spec = importlib.util.spec_from_file_location(modname, init_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[modname] = module
        spec.loader.exec_module(module)
        op = getattr(module, fname, None)

        return op

    def _load_operator_from_path(self, path: Union[str, Path], function: str, arg: List[Any], kws: Dict[str, Any]) -> Operator:
        """
        Load operator form local path.
        Args:
            path (`Union[str, Path]`):
                Path to the operator python file.
            arg (`List[str, Any]`):
                The init args for OperatorClass.
            kws (`Dict[str, Any]`):
                The init kwargs for OperatorClass.
        Returns
            (`typing.Any`)
                The `Operator` output.
        """
        path = Path(path)
        fname = function.split('/')[1].replace('-', '_')
        op_name = function.replace('-', '_').replace('/', '.')
        modname = 'towhee.operator.' + op_name

        all_pkg = [item.project_name for item in list(pkg_resources.working_set)]
        if 'requirements.txt' in (i.name for i in path.parent.iterdir()):
            with open(path.parent / 'requirements.txt', 'r', encoding='utf-8') as f:
                reqs = f.read().split('\n')
            for req in reqs:
                if not req:
                    continue
                pkg_name = re.split(r'(~|>|<|=|!| )', req)[0]
                pkg_name = pkg_name.replace('_', '-')
                if pkg_name not in all_pkg:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])

        op = self._load_op(modname, path, fname)
        if not op:
            engine_log.warning('The operator\'s format is outdated.')
            op = self._load_legacy_op(modname, path, fname)

        return self._instance_operator(op, arg, kws) if op is not None else None

    def _load_operator_from_hub(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:
        if '/' not in function:
            function = 'towhee/'+function
        try:
            path = get_operator(operator=function, tag=tag)
        except Exception as e:  # pylint: disable=broad-except
            err = '{}, {}'.format(str(e), traceback.format_exc())
            engine_log.error(err)
            return None

        return self._load_operator_from_path(path, function, arg, kws)

    def load_operator(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:
        """
        Attempts to load an operator from cache. If it does not exist, looks up the
        operator in a remote location and downloads it to cache instead. By standard
        convention, the operator must be called `Operator` and all associated data must
        be contained within a single directory.

        Args:
            function: (`str`)
                Origin and method/class name of the operator. Used to look up the proper
                operator in cache.
        """
        for factory in [self._load_operator_from_internal,
                        self._load_operator_from_registry,
                        self._load_operator_from_hub]:
            op = factory(function, arg, kws, tag)
            if op is not None:
                return op
        return None

    def _instance_operator(self, op, arg: List[Any], kws: Dict[str, Any]) -> Operator:
        if arg is None:
            arg = ()
        return op(*arg, **kws) if kws is not None else op(*arg)
