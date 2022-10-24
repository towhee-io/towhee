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

from typing import Dict, Any, Set, Tuple
from towhee.runtime.constants import (
    WindowConst,
    FilterConst,
    TimeWindowConst
)


def check_keys(info: Dict[str, Any], essentials: Set[str]):
    """
    Check if the src is a valid node dictionary to describe.

    Args:
        info (`Dict[str, Any]`): The info dictionary.
        essentials (`Set[str]`): The essential keys that node dictionary should contain.
    """
    info_keys = set(info.keys())
    if not isinstance(info, dict) or not essentials.issubset(info_keys):
        raise ValueError(f'Node {str(info)} is not valid, lack attr {essentials - info_keys}')


def check_set(inputs: Tuple, all_inputs: Set[str]):
    """
    Check if the inputs in all_inputs.

    Args:
        inputs (`Set[str]`): The inputs schema.
        all_inputs (`Set[str]`): The all inputs schema in the DAG util the node.
    """
    inputs = set(inputs)
    if not inputs.issubset(all_inputs):
        raise ValueError(f'The DAG Nodes inputs {str(inputs)} is not valid, which is not declared: {inputs - all_inputs}.')


def check_int(info: Dict[str, Any], checks: list):
    """
    Check if the info is type of int.

    Args:
        info (`Dict[str, Any]`): The essential set will be check.
        checks (`list`): The list to check.
    """
    for name in checks:
        if not isinstance(info[name], int):
            raise ValueError(f'The iteration param:{info} is not valid, the type of {name} is not int.')
        if info[name] <= 0:
            raise ValueError(f'The iteration param:{info} is not valid, the [{name}]<=0.')


def check_length(inputs: Tuple, outputs: Tuple):
    """
    Check if the length of inputs and outputs is equal.

    Args:
        inputs (`Tuple`): The inputs schema.
        outputs (`Tuple`): The outputs schema.
    """
    if len(inputs) != len(outputs):
        raise ValueError('The node is is not valid, the length of inputs if not equal to outputs.')


def check_node_iter(iter_type: str, iter_param: Dict[str, Any], inputs, outputs, all_inputs: Set[str]):
    """
    Check if the iteration info is valid.

    Args:
        iter_type (`str`): The type of the iteration, such as 'map', 'flat_map', 'filter', 'time_window'.
        iter_param (`Dict[str, any]`): The parameter for the iteration.
        inputs (`Tuple`): The inputs schema of the node.
        outputs (`Tuple`): The inputs schema of the node.
        all_inputs (`Set[str]`): The all inputs schema in the DAG util the node.
    """
    if iter_type == FilterConst.name:
        check_length(inputs, outputs)
        check_set(iter_param[FilterConst.param.filter_by], all_inputs)
    elif iter_type == TimeWindowConst.name:
        check_length(inputs, outputs)
        check_set(iter_param[TimeWindowConst.param.timestamp_col], all_inputs)
        check_int(iter_param, [TimeWindowConst.param.time_range_sec,
                               TimeWindowConst.param.time_step_sec])
    elif iter_type == WindowConst.name:
        check_length(inputs, outputs)
        check_int(iter_param, [WindowConst.param.size,
                               WindowConst.param.step])
