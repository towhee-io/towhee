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

    def _nnoperator_config(self, op, op_name, node_id, node):
        '''
        preprocess -> model -> postprocess
        '''
        models = []
        if hasattr(op, constant.PREPROCESS):
            model_name = '_'.join([node_id, op_name, 'preprocess'])
            converter = ProcessToTriton(op.preprocess, self._model_root,
                                        model_name, constant.PREPROCESS)            
            models.append({
                'model_name': model_name,
                'converter': converter,
                'inputs': converter.inputs,
                'outputs': converter.outputs
            })

        model_name = '_'.join([node_id, op_name, 'model'])
        converter = NNOpToTriton(op, self._model_root,
                                 model_name)
        models.append({
            'model_name': model_name,
            'converter': converter,
            'inputs': converter.inputs,
            'outputs': converter.outputs
        })
        if hasattr(op, constant.POSTPROCESS):
            model_name = '_'.join([node_id, op_name, 'postprocess'])
            converter = ProcessToTriton(op.postprocess, self._model_root,
                                        model_name, constant.POSTPROCESS)
            models.append({
                'model_name': model_name,
                'converter': converter,
                'inputs': converter.inputs,
                'outputs': converter.outputs
            })
        models[0]['id'] = node_id
        for i in range(1, len(models)):
            models[i]['id'] = models[i]['model_name']
        models[-1]['child_ids'] = node['child_ids']
        for i in range(len(models) - 1):
            models[i]['child_ids'] = [models[i + 1]['id']]
        return dict((model['id'], model) for model in models)

    def _pyop_config(self, op, node_id, node):
        model_name = node_id + '_' + node['op_name']
        converter = PyOpToTriton(op, self._model_root, model_name)
        config = {node_id: {
            'id': node_id,
            'model_name': model_name,
            'model_version': 1,
            'inputs': converter.inputs,
            'outputs': converter.outputs,
            'child_ids': node['child_ids']
        }}
        return config

    def _create_node_config(self, node_id, node):
        op_name = node['op_name']
        init_args = node['init_args']
        op = Builder._load_op(op_name, init_args)
        if op is None:
            return None

        if isinstance(op, NNOperator):
            return self._nnoperator_config(op, op_name, node_id, node)
        else:
            return self._pyop_config(op, node_id, node)

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
            self._runtime_dag.update(config)
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
