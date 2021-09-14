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

# builder = GraphBuilder()
# data1 = data0.map(op1)
# data2 = data1.batch(op2, bs=64)

import yaml
import requests
import os

from towhee.base_repr import BaseRepr
from towhee.dag.variable_repr import VariableRepr


class DataframeRepr(BaseRepr):
    """`DataframeRepr` represents a single dataframe within a graph.

    A single dataframe is composed of multiple individual variables, each of which is
    required in the next operator stage within the graph.

    Args:
        name:
            The representation name.
        src:
            The information of this dataframe.
    """
    def __init__(self):
        super().__init__()
        # `_columns` must be filled in after this representation is instantiated
        self._columns = []

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, value: VariableRepr, index: int):
        self._columns[index] = value

    @columns.setter
    def columns(self, value: VariableRepr):
        self._columns.append(value)

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
    def from_yaml(file_or_src: str):
        """Import a YAML file decribing this dataframe.

        Args:
            src:
                YAML file (pre-loaded as string) to import.
        """
        dataframe = DataframeRepr()

        src = dataframe.load_file(file_or_src)
        df = yaml.safe_load(src)

        dataframe.name = df['name']
        for col in df['columns']:
            dataframe.columns.append(VariableRepr(col['vtype'], col['dtype']))

        return dataframe
