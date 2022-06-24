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
import json
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

def create_ensemble(dag, name='pipeline', max_batch_size=128):
    # add parent_ids
    dag_dict = copy.deepcopy(dag)
    for i, j in dag_dict.items():
        dag_dict[i]['parent_ids'] = []
    for key, value in dag.items():
        if value['child_ids'] == []:
            for k, v in dag.items():
                if key in v['child_ids']:
                    dag_dict[key]['parent_ids'].append(k)
        else:
            for a, b in dag.items():
                if key in b['child_ids']:
                    dag_dict[key]['parent_ids'].append(a)
    dim_len_list = []
    for y, z in dag_dict.items():
        if z['parent_ids'] == []:
            for w, x in z['input'].items():
                dim_len_list.append(len(x[1]))
        elif z['child_ids'] == []:
            for u, v in z['output'].items():
                dim_len_list.append(len(v[1]))
        else:
            continue

    # add name/paltform/max_batch_size
    res_dict = dict()
    res_dict['name'] = name
    res_dict['platform'] = 'ensemble'
    res_dict['max_batch_size'] = max_batch_size

    # add input/output
    for m, n in dag_dict.items():
        if n['parent_ids'] == []:
            res_dict['input'] = []
            for x, y in n['input'].items():
                input = dict()
                input['name'] = x
                input['data_type'] = y[0]
                input['dims'] = y[1]
                res_dict['input'].append(input)
        elif n['child_ids'] == []:
            res_dict['output'] = []
            for p, q in n['output'].items():
                output = dict()
                output['name'] = p
                output['data_type'] = q[0]
                output['dims'] =q[1]
                res_dict['output'].append(output)

    # add ensemble_scheduling step
    # only allowed sequence ensemble scheduling
    res_dict['ensemble_scheduling'] = dict()
    res_dict['ensemble_scheduling']['step'] = []
    
    parent_id = []
    make_step(parent_id, dag_dict, res_dict)

    res_json = json.dumps(res_dict, sort_keys=False, indent=2, separators=(',', ': '))
    res_str = str(res_json)
    for i in range(10):
        input_mapk = 'input_map'+str(i)
        output_mapk = 'output_map'+str(i)
        res1 = res_str.replace(input_mapk, "input_map")
        res = res1.replace(output_mapk, 'output_map')
        res_str = res
    return move_mark(res, dim_len_list)

def make_step(parent_id, dag_dict, res_dict):
    for c, d in dag_dict.items():
        if d['parent_ids'] == parent_id:
            input_count = 0
            output_count = 0
            step = dict()
            step['model_name'] = d['model_name']
            step['model_version'] = d['model_version']

            for e, f in d['input'].items():
                input_map_name = 'input_map'+str(input_count)
                step[input_map_name] = dict()
                step[input_map_name]['key'] = e
                if d['parent_ids'] == []:
                    step[input_map_name]['value'] = e
                else:
                    parent_key = d['parent_ids'][0]
                    step[input_map_name]['value'] = str(dag_dict[parent_key]['model_name'])+'_'+str(d['parent_ids'][0])
                input_count = input_count + 1

            for g, h in d['output'].items():
                output_map_name = 'output_map'+str(output_count)
                step[output_map_name] = dict()
                step[output_map_name]['key'] = g
                if d['child_ids'] == []:
                    step[output_map_name]['value'] = g
                else:
                    step[output_map_name]['value'] = str(d['model_name'])+'_'+str(c)
                output_count = output_count + 1
            nlist = []
            nlist.append(d['id'])
            parent_id = nlist
            res_dict['ensemble_scheduling']['step'].append(step)
            del step
            if parent_id == []:
                return
            else:
                make_step(parent_id, dag_dict, res_dict)
        else:
            break

def move_mark(data, dim_lens):
    data_copy = copy.copy(data)
    res = []
    while True:
        index = data_copy.partition(":")
        x = index[0]
        if x == data_copy:
            res.append(x)
            res = ''.join(res)
            break
        for i in range(2):
            a = x.rfind('"')
            b = []
            b.append(x[:a])
            b.append(x[a+1:])
            x = ''.join(b)
        res.append(x)
        res.append(index[1])
        data_copy = index[2]

    # solve dims
    res_data = []
    dim = 0
    while True:
        s = res.partition("dims: ")
        y = s[0]
        if y == res:
            res_data.append(y)
            res_data = ''.join(res_data)
            break
        else:
            res_data.append(y)
            res_data.append(s[1])
            change = s[2].replace('\n       ', '', 1)
            res = change.replace('\n       ', '', dim_lens[dim]-1)
            res = res.replace('\n     ', '', 1)
            dim = dim + 1

    # solve :
    res_data = res_data.replace('input:', 'input')
    res_data = res_data.replace('output:', 'output')
    res_data = res_data.replace('step:', 'step')
    res_data = res_data.replace('map:', 'map')

    # solve comma
    res_data = res_data.replace(',\n', '\n')
    res_data = res_data.replace('{\n','', 1)
    res_data = res_data.replace('  ', '', 1)
    res_data = res_data.replace('\n}', '\n')
    res_data = res_data.replace('\n  ','\n')
    return res_data