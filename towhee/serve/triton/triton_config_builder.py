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


def create_modelconfig(model_name, max_batch_size, inputs, outputs, backend,
                       enable_dynamic_batching=False, preferred_batch_size=None,
                       max_queue_delay_microseconds=None,
                       instance_count=1, device_ids=None):
    '''
    example of input and output:
        {
            INPUT0': ('TYPE_INT8', [-1, -1, 3]),
            INPUT1': ('TYPE_FP32', [-1, -1, 3])
        }
    '''
    config = 'name: "{}"\n'.format(model_name)
    config += 'backend: "{}"\n'.format(backend)
    if max_batch_size is not None:
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
    if inputs is not None:
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
    if outputs is not None:
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

    instance_count = 1 if instance_count is None else instance_count
    if device_ids is not None:
        config += '''
instance_group [
    {{
        kind: KIND_GPU
        count: {}
        gpus: {}
    }}
]\n'''.format(instance_count, device_ids)
    else:
        config += '''
instance_group [
    {{
        kind: KIND_CPU
        count: {}
    }}
]\n'''.format(instance_count)

    return config
