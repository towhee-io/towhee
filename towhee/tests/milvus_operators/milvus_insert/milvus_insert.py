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
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
import numpy


class MilvusInsert(Operator):
    """
    Insert to Milvus
    """

    def __init__(self, host: str, port: str, c_name: str) -> None:
        super().__init__()
        self.connect = connections.connect(host=host, port=port)
        self._c_name = c_name

    def __call__(self, embs: numpy.ndarray) -> NamedTuple("Outputs", [("pk", list)]):
        self.connect
        (_,dim) = embs.shape
        COLLECTION_NAME = self._c_name

        if utility.has_collection(COLLECTION_NAME):
            #print('Collection exists: ', COLLECTION_NAME)
            my_collection = Collection(COLLECTION_NAME)
        else:
            my_fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="emb", dtype=DataType.FLOAT_VECTOR, dim=dim)
            ]
            my_schema = CollectionSchema(fields=my_fields, description="")
            my_collection = Collection(name=COLLECTION_NAME, schema=my_schema)
            print('Collection created: ', COLLECTION_NAME)
        mr = my_collection.insert([embs.tolist()])

        Outputs = NamedTuple("Outputs", [("pk", list)])
        return Outputs(mr.primary_keys)
