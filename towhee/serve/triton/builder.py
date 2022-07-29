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

from typing import Dict, List
import traceback
import logging

from towhee import ops
from towhee.operator import NNOperator
from towhee.serve.triton import constant
from towhee.serve.triton.to_triton_models import PreprocessToTriton, PostprocessToTriton, ModelToTriton, PyOpToTriton, EnsembleToTriton

logger = logging.getLogger()


class Builder:
    '''
    Build triton models from towhee pipeline.

    In V1, we only support a chain graph for we have not traced the input and output map.

    Args:
        dag (`dict`):
            Output of dc.compile_dag()

            example:  Only remain the info which is used in the TritonBuilder.
                {
                    'start': {
                         'op_name': 'dummy_input', 'init_args': None, 'child_ids': ['cb2876f3']
                    },
                    'cb2876f3': {
                         'op_name': 'local/triton_py', 'init_args': {}, 'child_ids': ['fae9ba13']
                    },
                    'fae9ba13': {
                         'op_name': 'local/triton_nnop', 'init_args': {'model_name': 'test'},'child_ids': ['end']
                    },
                    'end': {
                         'op_name': 'end', 'init_args': None, 'call_args': None, 'child_ids': []
                    }
                }
        model_root (`str`):
            Triton models root.

    '''
    def __init__(self, dag: Dict, model_root: str):
        self.dag = dag
        self._runtime_dag = None
        self._model_root = model_root
        self._ensemble_config = None

    def _nnoperator_config(self, op, op_name, node_id, node):
        '''
        preprocess -> model -> postprocess
        '''
        models = []
        op_config = node.get(constant.OP_CONFIG, None)
        if op_config is None:
            op_config = {}
        if hasattr(op, constant.PREPROCESS):
            model_name = '_'.join([node_id, op_name, 'preprocess']).replace('/', '_')
            converter = PreprocessToTriton(op, self._model_root,
                                           model_name, op_config)
            models.append({
                'model_name': model_name,
                'model_version': 1,
                'converter': converter,
                'input': converter.inputs,
                'output': converter.outputs
            })

        model_name = '_'.join([node_id, op_name, 'model']).replace('/', '_')
        converter = ModelToTriton(op, self._model_root,
                                  model_name, op_config)
        models.append({
            'model_name': model_name,
            'model_version': 1,
            'converter': converter,
            'input': converter.inputs,
            'output': converter.outputs
        })
        if hasattr(op, constant.POSTPROCESS):
            model_name = '_'.join([node_id, op_name, 'postprocess']).replace('/', '_')
            converter = PostprocessToTriton(op, self._model_root,
                                            model_name, op_config)
            models.append({
                'model_name': model_name,
                'model_version': 1,
                'converter': converter,
                'input': converter.inputs,
                'output': converter.outputs
            })
        models[0]['id'] = node_id
        for i in range(1, len(models)):
            models[i]['id'] = models[i]['model_name']
        models[-1]['child_ids'] = node['child_ids']
        for i in range(len(models) - 1):
            models[i]['child_ids'] = [models[i + 1]['id']]
        return dict((model['id'], model) for model in models)

    def _pyop_config(self, op: 'Operator', node_id: str, node: Dict) -> Dict:
        op_config = node.get(constant.OP_CONFIG, None)
        if op_config is None:
            op_config = {}
        model_name = node_id + '_' + node['op_name'].replace('/', '_')
        hub, name = node['op_name'].split('/')
        converter = PyOpToTriton(op, self._model_root, model_name,
                                 hub, name, node['init_args'], op_config)
        config = {node_id: {
            'id': node_id,
            'model_name': model_name,
            'model_version': 1,
            'converter': converter,
            'input': converter.inputs,
            'output': converter.outputs,
            'child_ids': node['child_ids']
        }}
        return config

    def _create_node_config(self, node_id: str, node: Dict):
        op_name = node['op_name']
        init_args = node['init_args']
        op = Builder._load_op(op_name, init_args)
        if op is None:
            logger.error('Load operator: [%s] by init args: [%s] failed', op_name, init_args)
            return None

        if isinstance(op, NNOperator):
            format_priority = node.get(constant.OP_CONFIG, {}).get(constant.FORMAT_PRIORITY, [])
            op_support_format = op.model.supported_formats if hasattr(op, 'model') and hasattr(op.model, 'supported_formats') else []
            if set(format_priority) & set(op_support_format):
                return self._nnoperator_config(op, op_name, node_id, node)
        return self._pyop_config(op, node_id, node)

    @staticmethod
    def _load_op(op_name: str, init_args: Dict) -> 'Operator':
        '''
        op_name:
            {hub_id}/{name}
        '''
        try:
            hub_id, name = op_name.split('/')
            hub = getattr(ops, hub_id)
            if not init_args:
                return getattr(hub, name)().get_op()
            else:
                return getattr(hub, name)(**init_args).get_op()
        except Exception as e:  # pylint: disable=broad-except
            err = f'Load operator: [{op_name}] failed, errs {e}, {traceback.format_exc()}'
            logger.error(err)
            return None

    def load(self) -> bool:
        self._runtime_dag = {}

        for node_id, node in self.dag.items():
            if node_id in ['start', 'end']:
                continue
            if node['op_name'] in ['start', 'end']:
                continue
            if 'end' in node['child_ids']:
                node['child_ids'].remove('end')
            config = self._create_node_config(node_id, node)
            if config is None:
                logger.error('Create node config failed')
                return False
            self._runtime_dag.update(config)
        return True

    def _build(self) -> bool:
        EnsembleToTriton(self._runtime_dag, self._model_root, 'pipeline', 0).to_triton()
        for _, info in self._runtime_dag.items():
            info['converter'].to_triton()
        return True

    def build(self):
        if self._runtime_dag is None:
            if not self.load():
                return False
        return self._build()


def main():
    import json  # pylint: disable=import-outside-toplevel
    import sys  # pylint: disable=import-outside-toplevel
    if len(sys.argv) != 4:
        sys.exit(-1)

    dag_file, model_root = sys.argv[1], sys.argv[2]
    with open(dag_file, 'rt', encoding='utf-8') as f:
        dag = json.load(f)
        if not Builder(dag, model_root).build():
            sys.exit(-1)
    sys.exit(0)


if __name__ == '__main__':
    main()
