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

import serve.triton.type_gen as tygen
from typing import List


def _from_tensor_to_obj(
    type_info: tygen.TypeInfo,
    obj_init_code: str,
    obj_var: str,
    tensor_vars: List[str]
):
    """
    Generate the codes converting Tensor to python object
    """
    trs_placeholders = [attr.tensor_placeholder for attr in type_info.attr_info]

    # check existence of placeholders
    for tr in trs_placeholders:
        if obj_init_code.find(tr) == -1:
            raise ValueError('Can not match placeholder %s with init function %s.' % (tr, trs_placeholders))

    if type_info.is_list:
        init_args = ['arg' + str(i) for i in range(len(trs_placeholders))]
        for tr, arg in zip(trs_placeholders, init_args):
            obj_init_code = obj_init_code.replace(tr, arg)

        line = obj_var + ' = [' + obj_init_code + ' for ' + ', '.join(init_args) + ' in zip(' + ', '.join(tensor_vars) + ')]'
    else:
        for tr, arg in zip(trs_placeholders, tensor_vars):
            obj_init_code = obj_init_code.replace(tr, arg)

        line = obj_var + ' = ' + obj_init_code

    return [line]


def _from_obj_to_tensor(
    type_info: tygen.TypeInfo,
    obj_var: str,
    tensor_vars: List[str],
    tensor_names: List[str],
    obj_placeholder='$obj'
):
    """
    Generate the codes converting python object to Tensor
    """

    lines = []

    for attr, tr_name, tr_var in zip(type_info.attr_info, tensor_names, tensor_vars):
        data = attr.obj_placeholder.replace(obj_placeholder, obj_var)
        dtype = attr.numpy_dtype
        ndarray = 'numpy.array(' + data + ', ' + dtype + ')'
        line = tr_var + ' = pb_utils.Tensor(\'' + tr_name + '\', ' + ndarray + ')'
        lines.append(line)

    return lines
