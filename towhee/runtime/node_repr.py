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

from typing import Dict, Any, Tuple, Union, Callable, List, Optional

from pydantic import BaseModel

from towhee.runtime.node_config import NodeConfig


class OperatorRepr(BaseModel):
    """
    OperatorRepr for operator representations.

    Args:
        operator (`Union[str, Callable]`): The operator, such as a callable (lambda, function) or the name of an op on the hub.
        type (`str`): The type of operator, such as 'hub', 'lambda' and 'callable'.
        init_args (`Tuple`): The args to initialize the operator.
        init_kws (`Dict[str, any]`): The kwargs to initialize the operator.
        tag (`str`): The tag for The function, defaults to 'main'.
    """

    operator: Union[None, str, Callable]
    type: Optional[str]
    init_args: Optional[Tuple]
    init_kws: Optional[Dict[str, Any]]
    tag: Optional[str] = 'main'
    latest: bool = False


class IterationRepr(BaseModel):
    """
    IterationRepr for iteration representations.

    Args:
        type (`str`): The type of the iteration, such as 'map', 'flat_map', 'filter', 'time_window'.
        param (`Dict[str, any]`): The parameter for the iteration, defaults to None.
    """
    type: str
    param: Optional[Dict[str, Any]] = None


class NodeRepr(BaseModel):
    """
    NodeRepr for node representations.

    Args:
        name (`str`): Name of the node, such as '_input', '_output' and id for the node.
        inputs (`Tuple`): Input schema to this node.
        outputs (`Tuple`): Output schema to this node.
        iter_info (`IterationRepr`): The iteration to this node.
        op_info (`OperatorRepr`): The operator to this node.
        config (`NodeConfig`): The configuration to this node.
        next_nodes (`List'): The next nodes.
        in_edges (`List`): The input edges about data queue schema, defaults to None.
        out_edges (`List`): The output edges about data queue schema, defaults to None.
    """
    uid: str
    inputs: Optional[Tuple]
    outputs: Optional[Tuple]
    iter_info: IterationRepr
    op_info: OperatorRepr
    config: NodeConfig
    next_nodes: Optional[List]
    in_edges: Optional[List] = None
    out_edges: Optional[List] = None

    @property
    def name(self):
        return self.config.name
