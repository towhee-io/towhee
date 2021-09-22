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
import logging
from typing import Dict, List

from towhee.dag.base_repr import BaseRepr
from towhee.dag.dataframe_repr import DataframeRepr


class OperatorRepr(BaseRepr):
    """This class encapsulates operator representations at compile-time.

    Args:
        name:
            Name of the operator represented by this object.
        df_in:
            Input dataframes(s) to this object.
        df_out:
            This operator's output dataframe.
    """
    def __init__(self):
        super().__init__()
        self._inputs = {}
        self._outputs = {}

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @inputs.setter
    def inputs(self, value: Dict[str, DataframeRepr]):
        self._inputs = value

    @outputs.setter
    def outputs(self, value: Dict[str, DataframeRepr]):
        self._outputs = value

    @staticmethod
    def is_valid(info: List[dict]) -> bool:
        """Check if the src is a valid YAML file to describe the operator(s) in Towhee.

        Args:
            info(`list`):
                The list loaded from the source file.
        """
        essentials = {'name', 'function', 'inputs', 'outputs', 'iterators'}
        if not isinstance(info, list):
            logging.error('src is not a valid YAML file')
            return False
        for i in info:
            if not isinstance(i, dict):
                logging.error('src is not a valid YAML file')
                return False
            if not essentials.issubset(set(i.keys())):
                logging.error('src cannot descirbe the operator(s) in Towhee')
                return False
        return True

    @staticmethod
    def from_yaml(file_or_src: str, dataframes: Dict[str, DataframeRepr]):
        """Import a YAML file decribing the operator(s).

        Args:
            file_or_src(`str`):
                YAML file (could be pre-loaded as string) to import.

            dataframes(`dict`):
                Dict with `DataframeRepr` objects as values to construct current
                `OperatorRepr`.

        Returns:
            The operators we described in `file_or_src`
        """
        operators = OperatorRepr.load_src(file_or_src)
        if not OperatorRepr.is_valid(operators):
            raise ValueError('file or src is not a valid YAML file to describe the operator in Towhee.')

        res = {}

        for op in operators:
            operator = OperatorRepr()
            operator.name = op['name']
            for ins in op['inputs']:
                operator.inputs[ins['name']] = {}
                operator.inputs[ins['name']]['df'] = dataframes[ins['df']]
                operator.inputs[ins['name']]['idx'] = ins['col']
            # output has only one dataframe
            outs = op['outputs'][0]
            operator.outputs[outs['name']] = {}
            operator.outputs[outs['name']]['df'] = dataframes[outs['df']]
            operator.outputs[outs['name']]['idx'] = outs['col']
            res[operator.name] = operator

        return res
