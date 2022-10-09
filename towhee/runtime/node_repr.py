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

from typing import Dict, Any, Tuple, Union, Callable, Set


class NodeRepr:
    """
    NodeRepr for node representations.

    Args:
        name (`str`): Name of the node, such as 'input', 'output' and id for the node.
        function (`Union[str, Callable]`): The operator, such as a callable (lambda, function) or the name of an op on the hub.
        init_args (`[str, Tuple]`): The args to initilize the operator.
        init_kws (`Dict[str, any]`): The kwargs to initilize the operator.
        inputs (`Union[str, Tuple]`): Input schema to this node.
        outputs (`Union[str, Tuple]`): Output schema to this node.
        fn_type (`str`): The of function, such as 'hub', 'lambda' and 'callable'.
        iteration (`str`): The iteration to this node, such as 'map', 'flat_map', 'filter', 'time_window'.
        config (`Dict[str, any]`): The configuration to this node.
        tag (`str`): The tag for The function, defauts to 'main'.
        param (`Dict[str, any]`): The parameter for the iteration, defaults to None.

    Examples:
    >>> from towhee.runtime.node_repr import NodeRepr
    >>> node_dict = {'inputs': None, 'outputs': ('a', 'b'), 'fn_type': '_input', 'iteration': 'map'}
    >>> nr = NodeRepr.from_dict('_input', node_dict)
    >>> print(nr.name, nr.function, nr.init_args, nr.init_kws, nr.inputs, nr.outputs, nr.fn_type, nr.iteration, nr.config, nr.tag, nr.param)
    input None None None None ('a', 'b') input map None main None
    """
    def __init__(
        self,
        name: str,
        function: Union[str, Callable],
        init_args: Tuple,
        init_kws: Dict[str, Any],
        inputs: Union[str, Tuple],
        outputs: Union[str, Tuple],
        fn_type: str,
        iteration: str,
        config: Dict[str, Any],
        tag: str = 'main',
        param: Dict[str, Any] = None,
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
        self._param = param

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
    def inputs(self) -> Union[str, Tuple]:
        return self._inputs

    @property
    def outputs(self) -> Union[str, Tuple]:
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
    def param(self) -> Dict[str, Any]:
        return self._param

    @staticmethod
    def is_valid(node: Dict[str, Any], essentials: Set[str]):
        """
        Check if the src is a valid node dictionary to describe.

        Args:
            node (`Dict[str, Any]`): The node dictionary.
            essentials (`Set[str]`): The essential keys that node dictionary should contain.

        Returns:
            (`bool | raise`)
                Return `True` if it is valid, else raise exception.
        """
        node_keys = set(node.keys())
        if not isinstance(node, dict) or not essentials.issubset(node_keys):
            raise ValueError(f'Node {str(node)} is not valid, lack attr {essentials - node_keys}')
        return True

    @staticmethod
    def from_dict(name: str, node: Dict[str, Any]) -> 'NodeRepr':
        """Return a NodeRepr from a description dict.

        Args:
            name (`str`): Name of the node, such as 'input', 'output' or id for the node.
            node (`Dict[str, Any]`): Dictionary about node info from dag.

        Returns:
            NodeRepr object.
        """
        if name in ['_input', '_output']:
            NodeRepr.is_valid(node, {'inputs', 'outputs', 'fn_type', 'iteration'})
            return NodeRepr(name, None, None, None, node['inputs'], node['outputs'], node['fn_type'],
                            node['iteration'], None, None, None)
        else:
            NodeRepr.is_valid(node, {'function', 'init_args', 'init_kws', 'inputs', 'outputs', 'fn_type', 'iteration', 'config', 'tag', 'param'})
            return NodeRepr(name, node['function'], node['init_args'], node['init_kws'], node['inputs'], node['outputs'], node['fn_type'],
                            node['iteration'], node['config'], node['tag'], node['param'])
