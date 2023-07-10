# Copyright 2023 Zilliz. All rights reserved.
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

from typing import Any
from towhee.runtime.factory import HubOp


class AnnSearch:
    """
    The ANN search operator is used to find the closest (or most similar)
    point to a given point in a given set, i.e. find similar embeddings.
    """

    faiss_index: HubOp = HubOp('ann_search.faiss_index')
    """
    Only for local test. If you want to use a vector database in a production environment,
    you can use Milvus(https://github.com/milvus-io/milvus).

    __init__(self, data_dir: str, top_k: int = 5)
        data_dir(`str`):
            Path to store data.
        top_k(`int`):
            top_k similar data

    __call__(self, query: 'ndarray') -> List[Tuple[id: int, score: float, meta: dict]
        query(`ndarray`):
            query embedding

    Example;

    .. code-block:: python

        from towhee import pipe, ops

        p = (
                pipe.input('vec')
                .flat_map('vec', 'rows', ops.ann_search.faiss_index('./data_dir', 5))
                .map('rows', ('id', 'score'), lambda x: (x[0], x[1]))
                .output('id', 'score')
            )

        p(<your-vector>)
    """

    milvus_client: HubOp = HubOp('ann_search.milvus_client')
    """
    Search embedding in Milvus, please make sure you have inserted data to Milvus Collection.

    __init__(self, host: str = 'localhost', port: int = 19530, collection_name: str = None,
                 user: str = None, password: str = None, **kwargs)
        host(`str`):
            The host for Milvus.
        port(`str`):
            The port for Milvus.
        collection_name(`str`):
            The collection name for Milvus.
        user(`str`)
            The user for Zilliz Cloud, defaults to None.
        password(`str`):
            The password for Zilliz Cloud, defaults to None.
        kwargs(`dict`):
            The same with pymilvus search: https://milvus.io/docs/search.md

    __call__(self, query: 'ndarray') -> List[Tuple]
        query(`ndarray`):
            query embedding

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        p = (
            pipe.input('text')
            .map('text', 'vec', ops.sentence_embedding.transformers(model_name='all-MiniLM-L12-v2'))
            .flat_map('vec', 'rows', ops.ann_search.milvus_client(host='127.0.0.1', port='19530',
                                                                  collection_name='text_db2', **{'output_fields': ['text']}))
            .map('rows', ('id', 'score', 'text'), lambda x: (x[0], x[1], x[2]))
            .output('id', 'score', 'text')
        )

        DataCollection(p('cat')).show()

    """

    milvus_multi_collections: HubOp = HubOp('ann_search.osschat_milvus')
    """
    `milvus_multi_collections <https://towhee.io/ann-search/osschat-milvus>`_ A client that can access multiple collections.

    __init__(self, host: str = 'localhost', port: int = 19530,
                 user: str = None, password: str = None, **kwargs):
        host(`str`):
            The host for Milvus.
        port(`str`):
            The port for Milvus.
        user(`str`)
            The user for Zilliz Cloud, defaults to None.
        password(`str`):
            The password for Zilliz Cloud, defaults to None.
        kwargs(`dict`):
            The same with pymilvus search: https://milvus.io/docs/search.md

    __call__(self, collection_name: str, query: 'ndarray') -> List[Tuple]
        collection_name(`str`):
            The collection name for Milvus.
        query(`ndarray`):
            query embedding

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        p = (
            pipe.input('text')
            .map('text', 'vec', ops.sentence_embedding.transformers(model_name='all-MiniLM-L12-v2'))
            .flat_map('vec', 'rows', ops.ann_search.milvus_multi_collections(host='127.0.0.1', port='19530', **{'output_fields': ['text']}))
            .map('rows', ('id', 'score', 'text'), lambda x: (x[0], x[1], x[2]))
            .output('id', 'score', 'text')
        )

        DataCollection(p('cat')).show()
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return HubOp('towhee.ann_search')(*args, **kwds)
