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

import json
from pathlib import Path

try:
    # pylint: disable=unused-import,ungrouped-imports
    import faiss
except ModuleNotFoundError as moduleNotFound:
    try:
        from towhee.utils.dependency_control import prompt_install
        prompt_install('faiss-cpu')
        # pylint: disable=unused-import,ungrouped-imports
        import faiss
    except:
        from towhee.utils.log import engine_log
        engine_log.error('faiss not found, you can install via `conda install -c conda-forge faiss-cpu` or `pip install faiss-cpu`.')
        raise ModuleNotFoundError('faiss not found, you can install via `conda install -c conda-forge faiss-cpu` or `pip install faiss-cpu`.') from \
            moduleNotFound


class KVStorage:
    '''
    Id maps of faiss.
    '''

    def __init__(self, fname=None):
        self._fname = fname
        if self._fname is not None and Path(self._fname).exists():
            with open(self._fname, encoding='utf-8') as f:
                self._kv = json.load(f)
        else:
            self._kv = {}

    def add(self, k, v):
        self._kv[str(k)] = v

    def get(self, k):
        return self._kv.get(str(k))

    def dump(self):
        with open(self._fname, 'w+', encoding='utf-8') as f:
            json.dump(self._kv, f)
