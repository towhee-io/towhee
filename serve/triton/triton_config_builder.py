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

from typing import Dict, List, Tuple, Any

import serve.triton.type_gen as tygen


class TritonModelConfigBuilder:
    """
    This class is used to generate a triton model config file.
    """

    def __init__(self, model_spec: Dict):
        self.model_spec = model_spec

    @staticmethod
    def _get_triton_schema(schema: List[Tuple[Any, Tuple]], var_prefix):

        type_info_list = tygen.get_type_info(schema)
        attr_info_list = [attr_info for type_info in type_info_list for attr_info in type_info.attr_info]

        schema = {}
        for i, attr_info in enumerate(attr_info_list):
            var = var_prefix + str(i)
            dtype = attr_info.triton_dtype
            shape = list(attr_info.shape)
            schema[var] = (dtype, shape)

        return schema

    @staticmethod
    def get_input_schema(schema: List[Tuple[Any, Tuple]]):
        return TritonModelConfigBuilder._get_triton_schema(schema, 'INPUT')

    @staticmethod
    def get_output_schema(schema: List[Tuple[Any, Tuple]]):
        return TritonModelConfigBuilder._get_triton_schema(schema, 'OUTPUT')


def create_modelconfig(model_name, max_batch_size, inputs, outputs, backend,
                       enable_dynamic_batching=None, preferred_batch_size=None,
                       max_queue_delay_microseconds=None):
    '''
    example of input and output:
        {
            INPUT0': ('TYPE_INT8', [-1, -1, 3]),
            INPUT1': ('TYPE_FP32', [-1, -1, 3])
        }
    '''
    config = 'name: "{}"\n'.format(model_name)
    config += 'backend: "{}"\n'.format(backend)
    config += 'max_batch_size: {}\n'.format(max_batch_size)
    if enable_dynamic_batching:
        config += '''
dynamic_batching {
'''
        if preferred_batch_size is not None:
            config += '''
    preferred_batch_size: {}
'''.format(preferred_batch_size)
        if max_queue_delay_microseconds is not None:
            config += '''
    max_queue_delay_microseconds: {}
'''.format(max_queue_delay_microseconds)
        config += '''
}\n'''
    for input_name in inputs.keys():
        data_type, shape = inputs[input_name]
        config += '''
input [
  {{
    name: \"{}\"
    data_type: {}
    dims: {}
  }}
]\n'''.format(input_name, data_type, shape)
    for output_name in outputs.keys():
        data_type, shape = outputs[output_name]
        config += '''
output [
  {{
    name: \"{}\"
    data_type: {}
    dims: {}
  }}
]\n'''.format(output_name, data_type, shape)
    return config
