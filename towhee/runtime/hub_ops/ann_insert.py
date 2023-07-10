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


class AnnInsert:
    """
    The ANN Insert Operator is used to insert embeddings and create ANN indexes for fast similarity searches.
    """

    faiss_index: HubOp = HubOp('ann_insert.faiss_index')
    """
    Insert data into faiss. Only for local test. If you want to use a vector database in a
    production environment, you can use Milvus(https://github.com/milvus-io/milvus).

    __init__(self, data_dir: str, dimension: int = None):
        data_dir(`str`):
            Path to store data.
        dimension(`int`):
            The dimension of embedding.

    __call__(self, vec: 'ndarray', *args):
        vec(`ndarray`):
            embedding
        *args(`Any`):
            meta data.

    Example:

    .. code-block:: python

        from glob import glob
        from towhee import ops, pipe

        p = (
            pipe.input('file_name')
            .map('file_name', 'img', ops.image_decode.cv2())
            .map('img', 'vec', ops.image_text_embedding.clip(model_name='clip_vit_base_patch32', modality='image'))
            .map('vec', 'vec', ops.towhee.np_normalize())
            .map(('vec', 'file_name'), (), ops.ann_insert.faiss_index('./faiss', 512))
            .output()
        )

        fs = glob('./images/*.jpg')

        for f in fs:
            p(f)

        # Ensure data is written to disk.
        p.flush()
    """

    milvus_client: HubOp = HubOp('ann_insert.milvus_client')
    """
    Insert data into Milvus collections. Please make sure you have
    `created Milvus Collection <https://milvus.io/docs/create_collection.md>`_ before loading the data.

    __init__(self, host: str, port: int, collection_name: str, user: str = None, password: str = None):
        host(`str`):
            The host for Milvus.
        port(`str`):
            The port for Milvus.
        collection_name(`str`):
            The collection name for Milvus.
        user(`str`)
            The user for Zilliz Cloud, defaults to None.
        password(`str`):
            he password for Zilliz Cloud, defaults to None.

    __call__(self, *data) -> 'pymilvus.MutationResult':
        data(`list`)
            The data to insert into milvus.

    Example:

    .. code-block:: python

        import towhee

        from towhee import ops

        p = (
                towhee.pipe.input('vec')
                .map('vec', (), ops.ann_insert.milvus_client(host='127.0.0.1', port='19530', collection_name='test_collection'))
                .output()
                )
        p(vec)
    """

    milvus_multi_collections: HubOp = HubOp('ann_insert.osschat_milvus')
    """
    `milvus_multi_collections <https://towhee.io/ann-insert/osschat-milvus>`_ A client that can access multiple collections.

    __init__(self, host: str, port: int, user: str = None, password: str = None):
        host(`str`):
            The host for Milvus.
        port(`str`):
            The port for Milvus.
        user(`str`)
            The user for Zilliz Cloud, defaults to None.
        password(`str`):
            he password for Zilliz Cloud, defaults to None.

    __call__(self, collection_name: str, *data) -> 'pymilvus.MutationResult':
        collection_name(`str`):
            collection_name
        data(`list`):
            The data to insert into milvus.

    Example:

    .. code-block:: python

        from towhee import ops, pipe

        p = (
                pipe.input('collection_name', 'vec')
                .map(('collection_name', 'vec'), (), ops.ann_insert.milvus_multi_collections(host='127.0.0.1', port='19530'))
                .output()
                )

        p(vec)
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return HubOp('towhee.ann_insert')(*args, **kwds)
