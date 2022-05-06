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

from pathlib import Path
import json

import faiss
import numpy as np
from towhee.connectors.ann_index import ANNIndex
from towhee.functional.entity import Entity


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


class FaissIndex(ANNIndex):
    '''
    FaissIndex base
    '''

    def __init__(self, param: str, dim=None, path: str = None):
        self._storage_dir = path
        self._dim = dim
        self._param = param
        self._kv_storage = KVStorage(self.kv_file)

    @property
    def index_file(self):
        if self._storage_dir is None:
            return None

        if not hasattr(self, '_index_file'):
            self._index_file = Path(self._storage_dir) / 'index.bin'
        return self._index_file

    @property
    def kv_file(self):
        if self._storage_dir is None:
            return None

        if not hasattr(self, '_kv_file'):
            self._kv_file = Path(self._storage_dir) / 'kv.bin'
        return self._kv_file

    def insert(self, k, e):
        # kid = np.array([abs(hash(k)) % (10 ** 8)]).astype(np.int64)
        kid = abs(hash(k)) % (10 ** 8)
        self._index.add_with_ids(e, np.array([kid]).astype(np.int64))
        self._kv_storage.add(kid, k)

    def search(self, e: 'ndarray', topk: int):
        scores, ids = self._index.search(e, topk)
        ids = ids[0].tolist()
        ret = []
        for i in range(len(ids)):
            k = self._kv_storage.get(ids[i])
            ret.append(Entity(**{'key': k, 'score': scores[0][i]}))
        return ret

    def _create_index(self):
        return faiss.index_factory(self._dim, self._param, faiss.METRIC_L2)

    def load(self):
        if self.index_file is None or not self.index_file.exists():
            self._index = self._create_index()
        else:
            self._index = faiss.read_index(str(self.index_file))

    def save(self):
        if self._storage_dir is not None:
            faiss.write_index(self._index, str(self.index_file))
            self._kv_storage.dump()

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.save()


class FaissIndexL2(FaissIndex):
    def __init__(self, dim, path=None):
        super().__init__('IDMap,Flat', dim, path)


class HNSW64Index(FaissIndex):
    def __init__(self, dim, path=None):
        super().__init__('IDMap,HNSW64', dim, path)
