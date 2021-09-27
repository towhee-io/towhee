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

from typing import Dict, List

from towhee.dag.base_repr import BaseRepr


class OperatorRepr(BaseRepr):
    """This class encapsulates operator representations at compile-time.

    Args:
        name:
            Name of the operator represented by this object.
        df_in:
            Input dataframes(s) to this object.
        df_out:
            This operator's output dataframe.
        iterator:
            This operator's iterator info.
    """
    def __init__(
        self,
        name: str,
        function: str,
        init_args: Dict[str, any],
        inputs: List[Dict[str, any]],
        outputs: List[Dict[str, any]],
        iter_info: Dict[str, any]
    ):
        super().__init__(name)
        self._function = function
        self._inputs = inputs
        self._outputs = outputs
        self._init_args = init_args
        self._iter_info = iter_info

    @property
    def function(self):
        return self._function

    @property
    def inputs(self) -> List:
        """
        Returns:
        [
            {
                'name': `arg name`,
                'df': `dataframe name`,
                'col': `col index`
            },
            ...
        ]
        """
        return self._inputs

    @property
    def outputs(self) -> List:
        """
        Returns:
        [
            {
                "df": `dataframe name`
            },
            ...
        ]
        """
        return self._outputs

    @property
    def init_args(self) -> Dict:
        """
        Returns:
        {
            `arg_name`: `arg value`,
            ...
        }
        """
        return self._init_args

    @property
    def iter_info(self):
        """
        Returns:
        {
            "type": `iter type`
        }
        """
        return self._iter_info

    @staticmethod
    def from_dict(info: Dict) -> 'OperatorRepr':
        """
        Args:
            info:
               Operator info
               {
                   'name': 'example_op',
                   'function': 'test_function',
                   'init_args': {
                       'arg1': 1,
                       'arg2': 'test'
                   },
                   'inputs': [
                       {
                           'df': 'input1',
                           'col': 0
                       },
                       {
                           'df': 'input2',
                           'col': 0
                       }
                   ],
                   'outputs': [
                       {
                           'df': 'input1'
                       },
                   ],
                   'iter_info': {
                       'type': 'map'
                   }
               }

            dataframes(`dict`):
                Dict with `DataframeRepr` objects as values to construct current
                `OperatorRepr`.

        Returns:
            The operators we described in `file_or_src`
        """
        if not BaseRepr.is_valid(info, {'name', 'init_args', 'function', 'inputs', 'outputs', 'iter_info'}):
            raise ValueError('Invalid operator info.')

        return OperatorRepr(info['name'], info['function'], info['init_args'], info['inputs'], info['outputs'], info['iter_info'])
