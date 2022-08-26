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

from towhee.engine import register

# pylint: disable=import-outside-toplevel
# pylint: disable=invalid-name

@register(name='builtin/milvus_search')
class milvus_search:  # pragma: no cover
    """
    Search for embedding vectors in Milvus. Note that the Milvus collection has data before searching,
    refer to DataCollection Mixin `to_milvus`.

    Args:
        collection (`str` or `pymilvus.Collection`):
            The collection name or pymilvus.Collection in Milvus.
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

    Examples:

    >>> import towhee
    >>> from pymilvus import connections
    >>> connections.connect(host='localhost', port='19530')
    >>> (
    ...    towhee.glob['path']('./*.jpg')
    ...           .image_decode['path', 'img']()
    ...           .image_embedding.timm['img', 'vec'](model_name='resnet50')
    ...           .milvus_search['vec', 'results'](collection='test')
    ...           .to_list()
    ... )
    [<Entity dict_keys(['path', 'img', 'vec', 'results'])>,
     <Entity dict_keys(['path', 'img', 'vec', 'results'])>]
    """
    def __init__(self, collection, **kwargs):
        from towhee.utils.thirdparty.milvus_utils import Collection

        if isinstance(collection, str):
            self.collection = Collection(collection)
        elif isinstance(collection, Collection):
            self.collection = collection
        self.kwargs = kwargs

        if 'anns_field' not in self.kwargs:
            fields_schema = self.collection.schema.fields
            for schema in fields_schema:
                if schema.dtype in (101, 100):
                    self.kwargs['anns_field'] = schema.name

        if 'limit' not in self.kwargs:
            self.kwargs['limit'] = 10

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

        if 'param' not in self.kwargs:
            if len(self.collection.indexes) != 0:
                index_type = self.collection.indexes[0].params['index_type']
                self.kwargs['param'] = index_params[index_type]
            elif 'metric_type' in self.kwargs:
                self.kwargs['param'] = {'metric_type': self.kwargs['metric_type']}
            else:
                self.kwargs['param'] = {'metric_type': 'L2'}

    def __call__(self, query: list):
        from towhee.functional.entity import Entity

        milvus_result = self.collection.search(
            data=[query],
            **self.kwargs
        )

        result = []
        for re in milvus_result:
            for hit in re:
                dicts = dict(id=hit.id, score=hit.score)
                if 'output_fields' in self.kwargs:
                    dicts.update(hit.entity._row_data)
                result.append(Entity(**dicts))
        return result
