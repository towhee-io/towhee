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

from typing import Dict, Any, Set, Tuple, Optional
from pydantic import BaseModel, constr, validator

from towhee.runtime.constants import (
    WindowConst,
    FilterConst,
    TimeWindowConst
)


# pylint: disable=no-self-argument
class IntForm(BaseModel):
    data: int

    @validator('data')
    def must_larger_than_zero(cls, v):
        if v <= 0:
            raise ValueError(f'The iteration param is not valid, the [{v}]<=0.')
        return v


class TupleForm(BaseModel):
    data: Optional[Tuple[str, ...]]
    schema_data: Optional[Tuple[constr(regex='^[a-zA-Z_][a-zA-Z_0-9]*$'), ...]]

    @validator('*', pre=True)
    def must_be_tuple(cls, v):
        if isinstance(v, str):
            return (v,)
        return v


class SetForm(BaseModel):
    data: Set[str]

    @validator('data', pre=True)
    def must_be_set(cls, v):
        if isinstance(v, str):
            return {v}
        if not isinstance(v, set):
            return set(v)
        return v


def check_set(inputs: Tuple, all_inputs: Set[str]):
    """
    Check if the inputs in all_inputs.

    Args:
        inputs (`Tuple[str]`): The inputs schema.
        all_inputs (`Set[str]`): The all inputs schema in the DAG util the node.
    """
    inputs = SetForm(data=inputs).data
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
        IntForm(data=info[name])


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
        check_set(iter_param[TimeWindowConst.param.timestamp_col], all_inputs)
        check_int(iter_param, [TimeWindowConst.param.time_range_sec,
                               TimeWindowConst.param.time_step_sec])
    elif iter_type == WindowConst.name:
        check_int(iter_param, [WindowConst.param.size,
                               WindowConst.param.step])
