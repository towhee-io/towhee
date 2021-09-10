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
from typing import Dict
import yaml

from towhee.base_repr import BaseRepr
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
    def __init__(
        self,
        name: str,
        dataframes: Dict[str, DataframeRepr],
        src: str = None,
    ):
        super().__init__(name)
        self._inputs = {}
        self._outputs = {}
        if src:
            self.from_yaml(src, dataframes)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @inputs.setter
    def inputs(self, value: Dict[str, DataframeRepr]):
        self._df_in = value

    @outputs.setter
    def outputs(self, value: Dict[str, DataframeRepr]):
        self._df_out = value

    def from_yaml(self, src: str, dataframes: Dict[str, DataframeRepr]):
        """Import a YAML file decribing this operator.

        Args:
            src:
                YAML file (pre-loaded as string) to import.

            dataframes:
                Dict with `DataframeRepr` objects as values to construct current
                `OperatorRepr`
        """
        op = yaml.safe_load(src)
        self._name = op['name']
        for ins in op['inputs']:
            self._inputs[ins['name']] = {}
            self._inputs[ins['name']]['df'] = dataframes[ins['df']]
            self._inputs[ins['name']]['idx'] = ins['col']
        # output has only one dataframe
        outs = op['outputs'][0]
        self._outputs[outs['name']] = {}
        self._outputs[outs['name']]['df'] = dataframes[outs['df']]
        self._outputs[outs['name']]['idx'] = outs['col']
