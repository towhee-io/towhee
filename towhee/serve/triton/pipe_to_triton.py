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

import shutil
import traceback

from towhee.serve.triton.triton_files import TritonFiles
from towhee.serve.triton.bls import PIPELINE_MODEL_FILE
from towhee.serve.triton.triton_config_builder import create_modelconfig
from towhee.serve.triton import constant

from towhee.utils.log import engine_log


class PipeToTriton:
    """
    Pipeline to triton models

    Use Triton python backend and auto_complete mode.
    """
    def __init__(self, dag_repr: 'dag_repr',
                 model_root: str,
                 model_name: str,
                 server_conf: int):
        self._dag_repr = dag_repr
        self._model_root = model_root
        self._model_name = model_name
        self._server_conf = server_conf
        self._triton_files = TritonFiles(self._model_root, self._model_name)

    def _create_model_dir(self) -> bool:
        self._triton_files.root.mkdir(parents=True, exist_ok=True)
        self._triton_files.model_path.mkdir(parents=True, exist_ok=True)
        return True

    def _prepare_config(self) -> bool:
        config_lines = create_modelconfig(
            model_name=self._model_name,
            max_batch_size=None,
            inputs=None,
            outputs=None,
            backend='python',
            enable_dynamic_batching=True,
            preferred_batch_size=None,
            max_queue_delay_microseconds=None,
            instance_count=self._server_conf.get(constant.PARALLELISM),
            device_ids=None
        )
        with open(self._triton_files.config_file, 'wt', encoding='utf-8') as f:
            f.writelines(config_lines)
            return True

    def _gen_bls_model(self) -> bool:
        try:
            shutil.copyfile(PIPELINE_MODEL_FILE, self._triton_files.python_model_file)
            return True
        except Exception as e:  # pylint: disable=broad-except
            engine_log.error('Create pipeline model file failed, err: [%s]', str(e))
            return False

    def _process_pipe(self) -> bool:
        from towhee.utils.thirdparty.dill_util import dill as pickle  # pylint: disable=import-outside-toplevel
        try:
            with open(self._triton_files.pipe_pickle_path, 'wb') as f:
                pickle.dump(self._dag_repr, f, recurse=True)
            return True
        except Exception as e:  # pylint: disable=broad-except
            err = '{}, {}'.format(str(e), traceback.format_exc())
            engine_log.error('Pickle pipeline to %s failed, err: [%s]', self._triton_files.pipe_pickle_path, err)
            return False

    def process(self) -> bool:
        if not self._create_model_dir() \
           or not self._prepare_config() \
           or not self._gen_bls_model() \
           or not self._process_pipe():
            engine_log.error('Pickle pipeline failed')
            return False
        engine_log.info('Pickle pipeline success')
        return True
