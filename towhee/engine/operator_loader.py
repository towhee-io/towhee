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
import pkg_resources
from pkg_resources import DistributionNotFound

from towhee.operator import Operator
from towhee.operator.nop import NOPOperator
from towhee.operator.concat_operator import ConcatOperator
from towhee.engine import LOCAL_OPERATOR_CACHE
from towhee.hub.file_manager import FileManager
from towhee.hparam import param_scope
from towhee.utils.log import engine_log

from .operator_registry import OperatorRegistry
# pylint: disable=broad-except


class OperatorLoader:
    """
    Wrapper class used to load operators from either local cache or a remote
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

    def _load_interal_op(self, op_name: str, arg: List[Any], kws: Dict[str, Any]):
        if op_name in ['_start_op', '_end_op']:
            return NOPOperator()
        elif op_name == '_concat':
            return ConcatOperator(*arg, **kws)
        else:
            # Not a interal operator
            return None

    def load_operator_from_internal(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        return self._load_interal_op(function, arg, kws)

    def load_operator_from_registry(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        op = OperatorRegistry.resolve(function)
        return self.instance_operator(op, arg, kws) if op is not None else None

    def load_operator_from_packages(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        try:
            module, fname = function.split('/')
            fname = fname.replace('-', '_')
            op_cls = ''.join(x.capitalize() or '_' for x in fname.split('_'))

            # module = '.'.join([module, fname, fname])
            module = '.'.join(['towheeoperator', '{}_{}'.format(module, fname), fname])
            op = getattr(importlib.import_module(module), op_cls)
            return self.instance_operator(op, arg, kws) if op is not None else None
        except Exception:  # pylint: disable=broad-except
            with param_scope() as hp:
                if hp().towhee.hub.use_pip(False):
                    return None  # TODO: download and install pip package from hub
                else:
                    return None

    def _load_legacy_op(self, modname, path, fname):
        # support old version operator API
        spec = importlib.util.spec_from_file_location(modname, path.resolve())
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
        init_path = path.parent / '__init__.py'
        if not init_path.is_file():
            return None
        spec = importlib.util.spec_from_file_location(modname, init_path.resolve())
        module = importlib.util.module_from_spec(spec)
        sys.modules[modname] = module
        spec.loader.exec_module(module)
        op = getattr(module, fname, None)

        return op

    def load_operator_from_path(self, path: Union[str, Path], function: str, arg: List[Any], kws: Dict[str, Any]) -> Operator:
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
        fname = path.stem
        op_name = function.replace('-', '_').replace('/', '.')
        modname = 'towhee.operator.' + op_name

        if 'requirements.txt' in (i.name for i in path.parent.iterdir()):
            with open(path.parent / 'requirements.txt', 'r', encoding='utf-8') as f:
                reqs = f.read().split('\n')
            for req in reqs:
                try:
                    pkg_resources.require(req)
                except DistributionNotFound:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])

        op = self._load_op(modname, path, fname)
        if not op:
            engine_log.warning('The operator\'s format is outdated.')
            op = self._load_legacy_op(modname, path, fname)

        return self.instance_operator(op, arg, kws) if op is not None else None

    def load_operator_from_cache(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        if '/' not in function:
            function = 'towhee/'+function
        try:
            fm = FileManager()
            path = fm.get_operator(operator=function, tag=tag)
        except ValueError as e:
            raise ValueError('operator {} not found!'.format(function)) from e

        if path is None:
            raise FileExistsError('Cannot find operator.')

        return self.load_operator_from_path(path, function, arg, kws)

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
        Raises:
            FileExistsError
                Cannot find operator.
        """

        for factory in [self.load_operator_from_internal,
                        self.load_operator_from_registry,
                        self.load_operator_from_packages,
                        self.load_operator_from_cache]:
            op = factory(function, arg, kws, tag)
            if op is not None:
                return op
        return None

    def instance_operator(self, op, arg: List[Any], kws: Dict[str, Any]) -> Operator:
        with param_scope() as hp:
            if hp().towhee.dry_run(False):
                return NOPOperator()
            else:
                return op(*arg, **kws) if kws is not None else op()
