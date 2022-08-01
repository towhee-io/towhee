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
import copy

import towhee.serve.triton.type_gen as tygen


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

    if device_ids is not None:
        if instance_count <= len(device_ids):
            instance_count = len(device_ids)
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


class EnsembleConfigBuilder:
    '''
    Create ensemble config. Currently, we just support a chain struct graph.

    Example of input dag:
    triton_dag = {
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
    def __init__(self, dag, model_name, max_batch_size):
        self._dag = dag
        self._model_name = model_name
        self._max_batch_size = max_batch_size
        self._head = None
        self._tail = None
        self._ready_node = set(['start'])

    def _process_dag(self):
        '''
        Find out head and tail
        add input_map and output_map for every node.
        '''
        for v in self._dag.values():
            if v['child_ids'] == []:
                self._tail = v
            if v['parent_ids'] == ['start']:
                self._head = v 
                self._ready_node.add(v['id'])
        self._head['input_map'] = [(input, input) for input in self._head['input'].keys()]
        self._tail['output_map'] = [(output, output) for output in self._tail['output'].keys()]

        # add output_map
        for name, v in self._dag.items():
            if name == self._tail['id']:
                continue
            output_map = []
            for output in v['output']:
                output_map.append((output, '_'.join([str(name), output])))
            v['output_map'] = output_map

        # add input_map
        all_keys = set(list(self._dag.keys()))
        all_keys.add('start')
        while True:
            for i, j in self._dag.items():
                if len(j['parent_ids']) == 0:
                    continue
                if 'input_map' in j:
                    continue
                if set(j['parent_ids']) < (self._ready_node):
                    input_map = []
                    if 'input_map' not in j:
                        if  'input_info' in j:
                            for input, value in j['input'].items():
                                for pid in j['parent_ids']:
                                    if pid == 'start':
                                        for output_id, output_value in self._head['input'].items():
                                            if value == output_value:
                                                input_map.append((input, output_id))
                                                break
                                    else:
                                        for output_id, output_value in self._dag[pid]['output'].items():
                                            if value == output_value:
                                                input_map.append((input, '_'.join([str(pid), output_id])))
                                                break
                            j['input_map'] = input_map
                            self._ready_node.add(i)
                        else:
                            for input, value in j['input'].items():
                                pid = j['parent_ids'][0]
                                for output_id, output_value in self._dag[pid]['output'].items():
                                    if value == output_value:
                                        input_map.append((input, '_'.join([str(pid), output_id])))
                                        break
                            j['input_map'] = input_map
                            self._ready_node.add(i)
            if self._ready_node == all_keys:
                break
        
        # change final output
        dag_value = copy.deepcopy(self._dag)
        for k, v in dag_value.items():
            for i in v['output_map']:
                sign_has_output = False
                for j in v['child_ids']:
                    for a in dag_value[j]['input_map']:
                        if a[1] == i[1]:
                            sign_has_output = True
                            real_value = v['output'][i[0]]
                            break
                if sign_has_output:
                    continue
                elif i[0] != i[1]:
                    m = len(self._tail['output_map'])
                    self._dag[k]['output_map'].append((i[0], 'OUTPUT'+str(m)))
                    self._dag[k]['output_map'].remove(i)
                    self._tail['output']['OUTPUT'+str(m)] = real_value
                    self._tail['output_map'].append(('OUTPUT'+str(m), 'OUTPUT'+str(m)))

    def _gen_header(self):
        config = 'name: "{}"\n'.format(self._model_name)
        config += 'platform: "ensemble"\n'
        config += 'max_batch_size: {}\n'.format(self._max_batch_size)
        for input_name in self._head['input'].keys():
            data_type, shape = self._head['input'][input_name]
            config += '''
input [
  {{
    name: \"{}\"
    data_type: {}
    dims: {}
  }}
]\n'''.format(input_name, data_type, shape)
        for output_name in self._tail['output'].keys():
            data_type, shape = self._tail['output'][output_name]
            config += '''
output [
  {{
    name: \"{}\"
    data_type: {}
    dims: {}
  }}
]\n'''.format(output_name, data_type, shape)
        return config

    def _gen_single_step(self, model_name, inputs, outputs, is_tail):
        config = '''    {{
      model_name: \"{}\"
      model_version: {}
'''.format(model_name, 1)

        for data in inputs:
            config += '''
      input_map {{
        key: \"{}\"
        value: \"{}\"
      }}
'''.format(data[0], data[1])

        for output in outputs:
            config += '''
      output_map {{
        key: \"{}\"
        value: \"{}\"
      }}
'''.format(output[0], output[1])

        config += '    }'
        if not is_tail:
            config += ',\n'
        else:
            config += '\n'
        return config

    def _gen_steps(self):
        config = '''
  step [\n'''
        items = list(self._dag.values())
        size = len(items)
        for i in range(size):
            step = items[i]
            config += self._gen_single_step(step['model_name'], step['input_map'], step['output_map'], i == size - 1)
        config += '  ]\n'
        return config

    def _gen_ensemble_scheduling(self):
        config = '''
ensemble_scheduling: {\n'''
        config += self._gen_steps()
        config += '}'
        return config

    def gen_config(self):
        self._process_dag()
        config = self._gen_header()
        config += self._gen_ensemble_scheduling()
        return config
