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

from urllib.parse import urlsplit

from towhee.connectors.ann_index import ANNIndex
from towhee.functional.entity import Entity
# pylint: disable=import-outside-toplevel


class MilvusDB(ANNIndex):
    """
    ANN index base class, implements __init__, insert and search.
    """

    def __init__(self, uri: str = None, host: str = 'localhost', port: int = 19530, collection_name: str = None):
        """
        Init MilvusDB class and get an existing collection.
        """
        from towhee.utils.milvus_utils import Collection
        self._uri = uri
        if uri:
            host, port, collection_name = self._parse_uri()
        self._host = host
        self._port = port
        self._collection_name = collection_name

        self.connect()
        self.collection = Collection(self._collection_name)

    @property
    def uri(self):
        return self._uri

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port

    @property
    def collection_name(self):
        return self._collection_name

    def _parse_uri(self):
        try:
            results = urlsplit(self._uri)
            host, port = results.netloc.split(':')
            collection = results.path.strip('/')
            return host, port, collection
        except ValueError as e:
            raise ValueError('The input uri is not match: \'tcp://<milvus-host>:<milvus-port>/<collection-name>\', '
                             'such as \'tcp://localhost:19530/my_collection\'') from e

    def insert(self, *args):
        """
        Insert data with `ann_insert`.

        Args:
        data (`list`):
            The data to insert into milvus.

        Returns:
            A MutationResult object contains `insert_count` represents how many and a `primary_keys` of primary keys.

        """
        data = []
        for a in args:
            data.append(a if isinstance(a, list) else [a])
        mr = self.collection.insert(data)
        return mr

    def search(self, *args, **kwargs):
        """
        Search for embedding vectors in Milvus. Note that the Milvus collection has data before searching.

        Args:
            data (`list`):
                The data to search in milvus.
            kwargs
                The kwargs with collection.search, refer to https://milvus.io/docs/v2.0.x/search.md#Prepare-search-parameters.
                And the `anns_field` defaults to the vector field name, `limit` defaults to 10, and `metric_type` in `param` defaults to 'L2'
                if there has no index(FLAT), and for default index `param`:
                    IVF_FLAT: {"params": {"nprobe": 10}},
                    IVF_SQ8: {"params": {"nprobe": 10}},
                    IVF_PQ: {"params": {"nprobe": 10}},
                    HNSW: {"params": {"ef": 10}},
                    IVF_HNSW: {"params": {"nprobe": 10, "ef": 10}},
                    RHNSW_FLAT: {"params": {"ef": 10}},
                    RHNSW_SQ: {"params": {"ef": 10}},
                    RHNSW_PQ: {"params": {"ef": 10}},
                    ANNOY: {"params": {"search_k": 10}}.

        Returns:
            An towhee.Entity list with the top_k results with `id`, `score` and `output_fields`. When `output_files=['path']`, will return like this:
            [<Entity dict_keys(['id', 'score', 'path'])>,
             <Entity dict_keys(['id', 'score', 'path'])>,
             ... ...
             <Entity dict_keys(['id', 'score', 'path'])>]
        """
        if 'anns_field' not in kwargs:
            fields_schema = self.collection.schema.fields
            for schema in fields_schema:
                if schema.dtype in (101, 100):
                    kwargs['anns_field'] = schema.name

        if 'limit' not in kwargs:
            kwargs['limit'] = 10

        index_params = {
            'IVF_FLAT': {'params': {'nprobe': 10}},
            'IVF_SQ8': {'params': {'nprobe': 10}},
            'IVF_PQ': {'params': {'nprobe': 10}},
            'HNSW': {'params': {'ef': 10}},
            'RHNSW_FLAT': {'params': {'ef': 10}},
            'RHNSW_SQ': {'params': {'ef': 10}},
            'RHNSW_PQ': {'params': {'ef': 10}},
            'IVF_HNSW': {'params': {'nprobe': 10, 'ef': 10}},
            'ANNOY': {'params': {'search_k': 10}}
        }

        if 'param' not in kwargs:
            if len(self.collection.indexes) != 0:
                index_type = self.collection.indexes[0].params['index_type']
                kwargs['param'] = index_params[index_type]
            elif 'metric_type' in kwargs:
                kwargs['param'] = {'metric_type': kwargs['metric_type']}
            else:
                kwargs['param'] = {'metric_type': 'L2'}

        data = []
        for arg in args:
            if isinstance(arg, list):
                data += arg
            else:
                data.append(arg)
        milvus_result = self.collection.search(
            data=data,
            **kwargs
        )

        result = []
        for re in milvus_result:
            for hit in re:
                dicts = dict(id=hit.id, score=hit.score)
                if 'output_fields' in kwargs:
                    dicts.update(hit.entity._row_data)  # pylint: disable=protected-access
                result.append(Entity(**dicts))
        return result

    def load(self):
        """
        Load data with milvus collection.
        """
        self.collection.load()

    def count(self):
        return self.collection.num_entities

    def connect(self):
        from towhee.utils.milvus_utils import connections
        if not connections.has_connection('default'):
            connections.connect(host=self._host, port=self._port)

    def disconnect(self):
        from towhee.utils.milvus_utils import connections
        connections.disconnect('default')

    def __enter__(self):
        """
        Execute when enter.
        """
        self.connect()
        self.load()

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Execute when exit.
        """
        self.load()
        self.disconnect()
