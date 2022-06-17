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


from typing import List, Dict, Tuple


def to_triton_schema(op_schema: List[Tuple], prefix: str):
    '''
    prefix:
       INPUT OR OUTPUT.
       triton_schema key = prefix + index

    Op schema into to triton schema info
      op schema:
        [(Image, (-1, -1, 3)), (int, ())]

      triton_schema:
        [
            INPUT0': ('INT8', [-1, -1, 3]),
            INPUT1': ('STR', []),
            INPUT2': ('INT32', [])
        ]
    '''
    triton_schema = []
    for item in op_schema:
        # op type to triton type
        pass
    return triton_schema


def create_modelconfig(model_name, max_batch_size, inputs, outputs,
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
    config += 'backend: "python"\n'
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
        print(input_name)
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


def create_ensemble(dag: Dict, name='pipeline', max_batch_size=128):
    '''
    Create triton enseble config.

    input dag:
    {
    0: {
        'id': 0,
        'mode_name': 'image_decode',
        'model_version': 1,
        'inputs': {
            'INPUT0': ('TYPE_STRING', []),
        },
        'outputs': {
            'OUTPUT0': ('TYPE_INT8', [-1, -1, 3]),
            'OUTPUT1': ('TYPE_STRING', []),
        },
        'child_ids': [1]
    },
    1: {
        'id': 1,
        'mode_name': 'clip_preprocess',
        'model_version': 1,
        'inputs': {
            'IUTPUT0': ('TYPE_INT8', [-1, -1, 3]),
            'IUTPUT1': ('TYPE_STRING', []),            
        },
        'outputs': {
            'OUTPUT0': ('TYPE_FP32', [-1, 3, 224, 224])
        },
        'child_ids': [2]
    },
    2: {
        'id': 2,
        'mode_name': 'clip_model',
        'model_version': 1,
        'inputs': {
            'IUTPUT0': ('TYPE_FP32', [-1, 3, 224, 224])
        },
        'outputs': {
            'OUTPUT0': ('TYPE_FP32', [-1, 512])
        },
        'child_ids': [3]        
    },
    3: {
        'id': 3,
        'mode_name': 'clip_postprocess',
        'model_version': 1,
        'inputs': {
            'IUTPUT0': ('TYPE_FP32', [-1, 512])
        },
        'outputs': {
            'OUTPUT0': ('TYPE_FP32', [512])
        },
        'child_ids': []
    }
}

    '''
    return ''


if __name__ == '__main__':
    data = create_modelconfig('test', 128,
                              {'input0': ('INT8', [-1, -1, 3]),
                               'input1': ('FL16', [-1, -1, 3])},
                              {'output0': ('INT8', [-1, -1, 3]),
                               'output1': ('FL16', [-1, -1, 3])}
                              )
    print(data)
