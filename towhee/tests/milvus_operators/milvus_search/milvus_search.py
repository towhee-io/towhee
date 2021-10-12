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


from typing import NamedTuple
from towhee.operator import Operator
from pymilvus import connections, Collection
import numpy


class MilvusSearch(Operator):
    """
    Search in Milvus
    """

    def __init__(self, host: str, port: str, c_name: str, index_type: str, metric_type: str, nlist: int, nprobe: int, topk: int) -> None:
        super().__init__()
        self.connect = connections.connect(host=host, port=port)
        self._c_name = c_name
        self._nprobe = nprobe
        self._topk = topk
        self._index = index_type
        self._metric = metric_type
        self._nlist = nlist


    def __call__(self, embs: numpy.ndarray) -> NamedTuple("Outputs", [("results", list)]):
        self.connect
        vecs = embs.tolist()
        #vecs = embs
        COLLECTION_NAME = self._c_name
        NPROBE = self._nprobe
        TOPK = self._topk
        INDEX_TYPE = self._index
        METRIC_TYPE = self._metric
        NLIST = self._nlist

        my_collection = Collection(name = COLLECTION_NAME)
        my_index = {"index_type": INDEX_TYPE, "params": {"nlist": NLIST}, "metric_type": METRIC_TYPE}
        my_collection.create_index(field_name="emb", index_params=my_index)

        my_collection.release()
        my_collection.load()

        search_params = {"metric_type": "L2", "params": {"nprobe": NPROBE}}
        res = my_collection.search(vecs, "emb", param=search_params, limit=TOPK)
        full_res = []
        for x in res:
            full_res.append((x.ids, x.distances))
        Outputs = NamedTuple("Outputs", [("results", list)])
        return Outputs(full_res)
