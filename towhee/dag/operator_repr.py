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

from typing import Dict, List, Any

from towhee.dag.base_repr import BaseRepr


class OperatorRepr(BaseRepr):
    """This class encapsulates operator representations at compile-time.

    Args:
        name (`str`):
            Name of the operator represented by this object.
        function (`str`):
            The path leads to the operator.
        init_args (`Dict[str, any]`):
            The args to initilize the operator.
        inputs (`List[Dict[str, Any]]`):
            Input dataframes(s) to this object.
        outputs (`List[Dict[str, Any]]`):
            This operator's output dataframe.
        iter_info (`Dict[str, Any]`):
            This operator's iterator info.
    """
    def __init__(
        self,
        name: str,
        function: str,
        init_args: Dict[str, Any],
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
        iter_info: Dict[str, Any],
        tag: str = 'main'
    ):
        super().__init__(name)
        self._function = function
        self._tag = tag
        self._inputs = inputs
        self._outputs = outputs
        self._init_args = init_args
        self._iter_info = iter_info

    @property
    def function(self):
        return self._function

    @property
    def inputs(self) -> List[dict]:
        """
        Returns:
            (`List[dict]`)
                The inputs of the operator.
        """
        return self._inputs

    @property
    def outputs(self) -> List:
        """
        Returns:
            (`List[dict]`)
                The outputs of the operator.
        """
        return self._outputs

    @property
    def init_args(self) -> Dict[str, Any]:
        """
        Returns:
            (`Dict[str, Any]`)
                The args to initilize the operator.
        """
        return self._init_args

    @property
    def tag(self) -> str:
        """
        Returns:
            (`str`)
                The tag to load of the operator.
        """
        return self._tag

    @property
    def iter_info(self) -> Dict[str, Any]:
        """
        Returns:
            (` Dict[str, Any]`)
                The operator's iterator info.
        """
        return self._iter_info

    @staticmethod
    def from_dict(info: Dict[str, Any]) -> 'OperatorRepr':
        """
        Generate a OperatorRepr from a description dict.

        Args:
            info (`Dict[str, Any]`):
                A dict to describe the Operator.

        Returns:
            (`towhee.dag.OperatorRepr`)
                The OperatorRepe object.
        """
        if not BaseRepr.is_valid(info, {'name', 'init_args', 'function', 'inputs', 'outputs', 'iter_info'}):
            raise ValueError('Invalid operator info.')

        if 'tag' not in info:
            info['tag'] = 'main'

        return OperatorRepr(info['name'], info['function'], info['init_args'], info['inputs'], info['outputs'], info['iter_info'], info['tag'])

    @staticmethod
    def from_ir(function: str, init_args: Dict[str, Any]) -> 'OperatorRepr':
        return OperatorRepr('', function, init_args, None, None, None)
