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
    def is_format(info: list):
        """Check if the src is a valid YAML file to describe the operator(s) in Towhee.

        Args:
            info(`list`):
                The list loaded from the source file.
        """
        essentials = {'name', 'function', 'inputs', 'outputs', 'iterators'}
        if not isinstance(info, list):
            raise ValueError('src is not a valid YAML file')
        for i in info:
            if not isinstance(i, dict):
                raise ValueError('src is not a valid YAML file')
            if not essentials.issubset(set(i.keys())):
                raise ValueError('src cannot descirbe the operator(s) in Towhee')

    @staticmethod
    def load_file(file: str) -> dict:
        """Load the operator(s) information from a local YAML file.

        Args:
            file:
                The file path.

        Returns:
            The list loaded from the YAML file that contains the operator(s) information.
        """
        with open(file, 'r', encoding='utf-8') as f:
            res = yaml.safe_load(f)
        if isinstance(res, dict):
            res = [res]
        OperatorRepr.is_format(res)
        return res

    @staticmethod
    def load_url(url: str) -> list:
        """Load the operator(s) information from a remote YAML file.

        Args:
            file:
                The url points to the remote YAML file.

        Returns:
            The list loaded from the YAML file that contains the operator(s) information.
        """
        src = requests.get(url).text
        res = yaml.safe_load(src)
        if isinstance(res, dict):
            res = [res]
        OperatorRepr.is_format(res)
        return res

    @staticmethod
    def load_str(string: str) -> list:
        """Load the operator(s) information from a YAML file (pre-loaded as string).

        Args:
            string:
                The string pre-loaded from a YAML.

        Returns:
            The list loaded from the YAML file that contains the operator(s) information.
        """
        res = yaml.safe_load(string)
        if isinstance(res, dict):
            res = [res]
        OperatorRepr.is_format(res)
        return res

    @staticmethod
    def load_src(file_or_src: str) -> str:
        """Load the information for the representation.

        We support file from local file/HTTP/HDFS.

        Args:
            file_or_src(`str`):
                The source YAML file or the URL points to the source file or a str
                loaded from source file.

        returns:
            The YAML file loaded as list.
        """
        # If `file_or_src` is a loacl file
        if os.path.isfile(file_or_src):
            return OperatorRepr.load_file(file_or_src)
        # If `file_or_src` from HTTP
        elif file_or_src.lower().startswith('http'):
            return OperatorRepr.load_url(file_or_src)
        # If `file_or_src` is neither a file nor url
        return OperatorRepr.load_str(file_or_src)

    @staticmethod
    def from_yaml(file_or_src: str, dataframes: Dict[str, DataframeRepr]):
        """Import a YAML file decribing the operator(s).

        Args:
            file_or_src(`str`):
                YAML file (could be pre-loaded as string) to import.

            dataframes(`dict`):
                Dict with `OperatorRepr` objects as values to construct current
                `OperatorRepr`.

        returns:
            The operators we described in `file_or_src`
        """
        operators = OperatorRepr.load_src(file_or_src)

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
