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

from typing import Dict

from towhee.utils.log import engine_log
from towhee.runtime import ops, AcceleratorConf
from towhee.operator import NNOperator
from towhee.serve.triton import constant
from towhee.serve.triton.model_to_triton import ModelToTriton
from towhee.serve.triton.pipe_to_triton import PipeToTriton


class Builder:
    """
    Build triton models from towhee pipeline.
    """

    def __init__(self, dag_repr: 'DAGRepr', model_root: str, conf: Dict):
        self.dag_repr = dag_repr
        self._server_conf = conf
        self._model_root = model_root

    @staticmethod
    def _load_op(op_name: str, init_args: tuple, init_kws: Dict) -> 'Operator':
        """
        load operator
        """
        name = op_name.split('/')
        if len(name) == 2:
            hub_id, name = name[0], name[1]
        else:
            hub_id, name = 'towhee', name[0]
        try:
            hub = getattr(ops, hub_id)
            init_args = () if init_args is None else init_args
            init_kws = {} if init_kws is None else init_kws
            return getattr(hub, name)(*init_args, **init_kws).get_op()
        except Exception as e:  # pylint: disable=broad-except
            engine_log.error('Load operator: %s by init args: [%s] and kws: [%s] failed, error: %s', op_name, init_args, init_kws, e)
            return None

    def _model_convert(self, node_repr: 'NodeRepr'):
        op_name = node_repr.op_info.operator
        op_init_args = node_repr.op_info.init_args
        op_init_kws = node_repr.op_info.init_kws
        op = self._load_op(op_name, op_init_args, op_init_kws)
        if op is None:
            return False

        if isinstance(op, NNOperator):
            if not hasattr(op, 'supported_formats'):
                return True
            model_name = node_repr.name.replace('/', '.')
            converter = ModelToTriton(self._model_root, op, model_name, node_repr.config, self._server_conf)
            status = converter.to_triton()
            if status == -1:
                return False
            elif status == 0:
                return True
            inputs, outputs = converter.get_model_in_out()
            acc_conf = {'model_name': model_name, 'inputs': inputs, 'outputs': outputs}
            node_repr.config.acc_conf = AcceleratorConf(acc_type='triton', conf=acc_conf)
        return True

    def build(self) -> bool:
        for node_id, node in self.dag_repr.nodes.items():
            if node_id in ['_input', '_output'] or node.op_info.type != 'hub':
                continue
            if not self._model_convert(node):
                return False
        return PipeToTriton(self.dag_repr, self._model_root, constant.PIPELINE_NAME).process()


# pylint: disable=import-outside-toplevel
def main():
    from towhee.utils.thirdparty.dail_util import dill as pickle
    import sys
    import json
    if len(sys.argv) != 4:
        engine_log.error('The args is invalid, it must has three parameters in [dag_file_path, mode_root_dir, config_file_path]')
        sys.exit(-1)

    dag_file, model_root, config_file = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(dag_file, 'rb') as f_dag, open(config_file, 'rb') as f_conf:
        dag_repr = pickle.load(f_dag)
        config = json.load(f_conf)
        if not Builder(dag_repr, model_root, config).build():
            sys.exit(-1)
    sys.exit(0)


if __name__ == '__main__':
    main()
