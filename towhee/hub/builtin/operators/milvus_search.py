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
class milvus_search:
    """
    Search for embedding vectors in Milvus. Note that the Milvus collection has data before searching,
    refer to DataCollection Mixin `to_milvus`.

    Args:
        collection (`str` or `pymilvus.Collection`):
            The collection name or pymilvus.Collection in Milvus.
        kwargs
            The kwargs with collection.search, refer to https://milvus.io/docs/v2.0.x/search.md#Prepare-search-parameters.

    Examples:

    >>> import towhee
    >>> from pymilvus import connections
    >>> connections.connect(host='localhost', port='19530')
    >>> search_args = dict(
    ...    param={"metric_type": "L2", "params": {"nprobe": 10}},
    ...    output_fields=["count", "random_value"],
    ...    limit=10
    ... )
    >>> (
    ...    towhee.glob['path']('./*.jpg')
    ...           .image_decode['path', 'img']()
    ...           .image_embedding.timm['img', 'vec'](model_name='resnet50')
    ...           .milvus_search['vec', 'results'](collection='test', **search_args)
    ...           .to_list()
    ... )
    [<Entity dict_keys(['path', 'img', 'vec', 'results'])>,
     <Entity dict_keys(['path', 'img', 'vec', 'results'])>]
    """
    def __init__(self, collection, **kwargs):
        from towhee.utils.milvus_utils import Collection

        if isinstance(collection, str):
            self.collection = Collection(collection)
        elif isinstance(collection, Collection):
            self.collection = collection
        self.kwargs = kwargs

    def __call__(self, query: list):
        from towhee.functional.entity import Entity

        if 'anns_field' not in self.kwargs:
            fields_schema = self.collection.schema.fields
            for schema in fields_schema:
                if schema.dtype in (101, 100):
                    self.kwargs['anns_field'] = schema.name

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


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)
