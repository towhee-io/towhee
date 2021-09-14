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
import requests
import os

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
    def load_file(file_or_src: str):
        # we support file from local file/HTTP/HDFS
        # if `file_or_src` is a loacl file
        if os.path.isfile(file_or_src):
            with open(file_or_src, 'r', encoding='utf-8') as f:
                src = yaml.safe_dump(yaml.safe_load(f), default_flow_style=False)
            return src
        # if `file_or_src` from HTTP
        elif file_or_src.lower().startswith('http'):
            src = requests.get(file_or_src).content.decode('utf-8')
            return src
        # if `file_or_src` is YAMl (pre-loaded as str)
        return file_or_src

    @staticmethod
    def from_yaml(file_or_src: str, dataframes: Dict[str, DataframeRepr]):
        """Import a YAML file decribing this operator.

        Args:
            src:
                YAML file (pre-loaded as string) to import.

            dataframes:
                Dict with `DataframeRepr` objects as values to construct current
                `OperatorRepr`
        """
        operator = OperatorRepr()

        src = operator.load_file(file_or_src)
        op = yaml.safe_load(src)

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

        return operator
