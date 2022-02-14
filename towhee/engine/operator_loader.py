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
from typing import Any, Dict, Union

from towhee.operator import Operator
from towhee.operator.nop import NOPOperator
from towhee.operator.concat_operator import ConcatOperator
from towhee.engine import LOCAL_OPERATOR_CACHE

from .operator_registry import OperatorRegistry
from towhee.hub.file_manager import FileManager

from towhee.hparam import param_scope


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

    def load_operator_from_internal(
            self, function: str, args: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        return self._load_interal_op(function, args)

    def load_operator_from_registry(
            self, function: str, args: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        op = OperatorRegistry.resolve(function)
        return self.instance_operator(op, args) if op is not None else None

    def load_operator_from_packages(
            self, function: str, args: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        try:
            module, fname = function.split('/')
            fname = fname.replace('-', '_')
            op_cls = ''.join(x.capitalize() or '_' for x in fname.split('_'))

            # module = '.'.join([module, fname, fname])
            module = '.'.join(
                ['towheeoperator', '{}_{}'.format(module, fname), fname])
            op = getattr(importlib.import_module(module), op_cls)
            return self.instance_operator(op, args) if op is not None else None
        except Exception:  # pylint: disable=broad-except
            with param_scope() as hp:
                if hp().towhee.hub.use_pip(False):
                    return None  # TODO: download and install pip package from hub
                else:
                    return None

    def load_operator_from_path(self, path: Union[str, Path], args: Dict[str, Any]) -> Operator:
        """
        Load operator form local path.
        Args:
            path (`Union[str, Path]`):
                Path to the operator python file.
            args (`Dict[str, Any]`):
                The init args for OperatorClass.
        Returns
            (`typing.Any`)
                The `Operator` output.
        """
        path = Path(path)
        fname = Path(path).stem
        modname = 'towhee.operator.' + fname
        spec = importlib.util.spec_from_file_location(modname, path.resolve())

        # Create the module and then execute the module in its own namespace.
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Instantiate the operator object and return it to the caller for
        # `load_operator`. By convention, the operator class is simply the CamelCase
        # version of the snake_case operator.
        op_cls = ''.join(x.capitalize() or '_' for x in fname.split('_'))
        op = getattr(module, op_cls)
        return self.instance_operator(op, args) if op is not None else None

    def load_operator_from_cache(
            self, function: str, args: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        fm = FileManager()
        path = fm.get_operator(operator=function, tag=tag)

        if path is None:
            raise FileExistsError('Cannot find operator.')

        return self.load_operator_from_path(path, args)

    def load_operator(self, function: str, args: Dict[str, Any],
                      tag: str) -> Operator:
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

        for factory in [
                self.load_operator_from_internal,
                self.load_operator_from_registry,
                self.load_operator_from_packages,
                self.load_operator_from_cache
        ]:
            op = factory(function, args, tag)
            if op is not None:
                return op
        return None

    def instance_operator(self, op, args: Dict[str, Any]) -> Operator:
        with param_scope() as hp:
            if hp().towhee.dry_run(False):
                return NOPOperator()
            else:
                return op(**args) if args is not None else op()
