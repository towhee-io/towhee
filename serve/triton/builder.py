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

import traceback
import logging

from towhee import ops
from towhee.operator import NNOperator
from serve.triton import constant
from serve.triton.to_triton_models import ProcessToTriton, NNOpToTriton, PyOpToTriton

logger = logging.getLogger()


class Builder:
    '''
    Build triton models from towhee pipeline.

    In V1, we only support a chain graph for we have not traced the input and output map.
    '''
    def __init__(self, dag, model_root):
        self.dag = dag
        self._runtime_dag = None
        self._model_root = model_root
        self._ensemble_config = {}

    def _create_node_config(self, node_id, node):
        op_name = node['op_name']
        init_args = node['init_args']
        op = Builder._load_op(op_name, init_args)
        if op is None:
            return None

        new_id = op_name + '_' + node_id
        converter = {}
        if isinstance(op, NNOperator):
            converter['model'] = NNOpToTriton(op, self._model_root, new_id + '_model')
            if hasattr(op, constant.PREPROCESS):
                converter['preprocess'] = ProcessToTriton(op.preprocess, self._model_root, new_id + '_preprocess', constant.PREPROCESS)
            if hasattr(op, constant.POSTPROCESS):
                converter['postprocess'] = ProcessToTriton(op.postprocess, self._model_root, new_id + '_postprocess', constant.POSTPROCESS)
        else:
            converter['model'] = PyOpToTriton(op, self._model_root, new_id + '_model')
        config = {}
        config['op_name'] = op_name
        config['child_ids'] = node['child_ids']
        config['converter'] = converter
        return config

    @staticmethod
    def _load_op(op_name, init_args) -> 'Operator':
        try:
            hub_id, name = op_name.split('/')
            hub = getattr(ops, hub_id)
            if not init_args:
                return getattr(hub, name)().get_op()
            else:
                return getattr(hub, name)(**init_args).get_op()
        except Exception as e:
            err = f'Load operator: [{op_name}] failed, errs {e}, {traceback.format_exc()}'
            logger.error(err)
            return None

    def load(self) -> bool:
        self._runtime_dag = {}
        for node_id, node in self.dag.items():
            if node_id in ['start', 'end']:
                continue
            config = self._create_node_config(node_id, node)
            if config is None:
                return False
            self._runtime_dag[node_id] = config
        return True

    def _build(self):
        for node_id, info in self._runtime_dag.items():
            pass
        return True

    def build(self):
        if self._runtime_dag is None:
            if not self.load():
                return False
        return self._build()


# if __name__ == '__main__':
#     test_dag = {'start': {'op': 'stream', 'op_name': 'dummy_input', 'is_stream': False, 'init_args': None, 'call_args': {'*arg': (), '*kws': {}}, 'parent_ids': [], 'child_ids': ['cb2876f3']}, 'cb2876f3': {'op': 'map', 'op_name': 'towhee/image-decode', 'is_stream': True, 'init_args': {}, 'call_args': {'*arg': None, '*kws': {}}, 'parent_ids': ['start'], 'child_ids': ['fae9ba13']}, 'fae9ba13': {'op': 'map', 'op_name': 'towhee/clip_image', 'is_stream': True, 'init_args': {'model_name': 'clip_vit_b32'}, 'call_args': {'*arg': None, '*kws': {}}, 'parent_ids': ['cb2876f3'], 'child_ids': ['end']}, 'end': {'op': 'end', 'op_name': 'end', 'init_args': None, 'call_args': None, 'parent_ids': ['fae9ba13'], 'child_ids': []}}
#     builer = Builder(test_dag, './')
#     assert builer.load() is True
#     print(builer._runtime_dag)
