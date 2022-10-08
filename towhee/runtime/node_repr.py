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

from typing import Dict, List, Any, Tuple, Union


class NodeRepr:
    def __init__(
        self,
        name: str,
        function: str,
        init_args: Tuple,
        init_kws: Dict[str, Any],
        inputs: Union[str, Tuple],
        outputs: Union[str, Tuple],
        fn_type: str,
        iteration: str,
        config: Dict[str, Any],
        tag: str = 'main',
    ):
        self._name = name
        self._function = function
        self._init_args = init_args
        self._init_kws = init_kws
        self._inputs = inputs
        self._outputs = outputs
        self._fn_type = fn_type
        self._iteration = iteration
        self._config = config
        self._tag = tag

    @property
    def name(self):
        return self._name

    @property
    def function(self):
        return self._function

    @property
    def init_args(self) -> Tuple:
        return self._init_args

    @property
    def init_kws(self) -> Dict:
        return self._init_kws

    @property
    def input(self) -> Union[str, Tuple]:
        return self._inputs

    @property
    def output(self) -> Union[str, Tuple]:
        return self._outputs

    @property
    def fn_type(self) -> str:
        return self._fn_type

    @property
    def iteration(self) -> str:
        return self._iteration

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def input_schema(self) -> List:
        return self._function.input_schema()

    @property
    def output_schema(self) -> List:
        return self._function.output_schema()

    @staticmethod
    def from_dict(name: str, node: Dict[str, Any]) -> 'NodeRepr':
        if 'tag' not in node:
            node['tag'] = 'main'
        if name in ['input', 'output']:
            return NodeRepr(name, None, None, None, node['input'], node['output'], node['fn_type'],
                            node['iteration'], None, node['tag'])
        else:
            return NodeRepr(name, node['fn'][0], node['fn'][1], node['fn'][2], node['input'], node['output'], node['fn_type'],
                            node['iteration'], node['config'], node['tag'])
