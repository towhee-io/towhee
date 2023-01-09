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

# pylint: disable = import-outside-toplevel

from towhee.runtime.pipeline import Pipeline
from towhee.runtime.factory import ops


def get_image(x):
    from towhee.types import Image
    return Image(x.__array__(), 'RGB')


def merge_ndarray(x):
    import numpy as np
    return np.concatenate(x).reshape(-1, x[0].shape[0])


class InsertLevelDB:
    """
    Insert data into leveldb.
    """
    def __init__(self, path):
        self._path = path

    def __call__(self, key, val):
        from io import BytesIO
        import numpy as np
        from towhee.utils.thirdparty.plyvel_utils import plyvel
        path = self._path

        db = plyvel.DB(path, create_if_missing=True)

        if isinstance(val, np.ndarray):
            np_bytes = BytesIO()
            np.save(np_bytes, val, allow_pickle=True)
            val = np_bytes.getvalue()
        else:
            val = str(val).encode('utf-8')

        db.put(str(key).encode('utf-8'), val)
        db.close()

        return True


class FromLevelDB:
    """
    Load data from leveldb.
    """
    def __init__(self, path):
        self._path = path

    def __call__(self, keys):
        from io import BytesIO
        import numpy as np
        from towhee.utils.thirdparty.plyvel_utils import plyvel
        path = self._path

        db = plyvel.DB(path, create_if_missing=True)

        if isinstance(keys, str):
            vals = db.get(str(keys).encode('utf-8'))
            vals = BytesIO(vals)
            vals = np.load(vals, allow_pickle=True)
        else:
            vals = []
            for key in keys:
                val = db.get(str(key).encode('utf-8'))
                val = BytesIO(val)
                val = np.load(val, allow_pickle=True)
                vals.append(val)

        db.close()
        return vals


def create_milvus_collection(pipe_config):
    from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

    connections.connect(host=pipe_config.host, port=pipe_config.port)

    if utility.has_collection(pipe_config.collection_name):
        utility.drop_collection(pipe_config.collection_name)

    fields = [
        FieldSchema(name='id',
                    dtype=DataType.INT64,
                    descrition='the id of the embedding',
                    is_primary=True,
                    auto_id=True),
        FieldSchema(name='path',
                    dtype=DataType.VARCHAR,
                    descrition='the path of the embedding',
                    max_length=500),
        FieldSchema(name='embedding',
                    dtype=DataType.FLOAT_VECTOR,
                    descrition='video embedding vectors',
                    dim=pipe_config.dim)
    ]
    schema = CollectionSchema(fields=fields, description='video copy detection')
    collection = Collection(name=pipe_config.collection_name, schema=schema)

    index_params = {
        'metric_type': 'IP',
        'index_type': 'IVF_FLAT',
        'params': {
            'nlist': 1
        }
    }
    collection.create_index(field_name='embedding', index_params=index_params)

    return collection


def video_copy_detection_insert(pipe_config):
    p = (
        Pipeline.input('url')
            .flat_map('url', 'frames', ops.video_decode.ffmpeg(sample_type=pipe_config.sample_type, args={'time_step': pipe_config.time_step}))
            .map('frames', 'img', get_image)
            .map('img', 'emb', ops.image_embedding.timm(model_name=pipe_config.model_name))
            .window_all('emb', 'video_emb', merge_ndarray)
            .map(('url', 'emb'), 'insert_res', ops.ann_insert.milvus(uri=pipe_config.milvus_uri))
            .map(('url', 'video_emb'), ('url_vec_status'), InsertLevelDB(pipe_config.leveldb_path))
            .output('url', 'emb', 'video_emb')
    )

    return p


def video_copy_detection_search(pipe_config):
    p = (
        Pipeline.input('url')
            .flat_map('url', 'frames', ops.video_decode.ffmpeg(sample_type=pipe_config.sample_type, args={'time_step': pipe_config.time_step}))
            .map('frames', 'img', get_image)
            .map('img', 'emb', ops.image_embedding.timm(model_name=pipe_config.model_name))
            .window_all('emb', 'video_emb', merge_ndarray)
            .map('emb', 'res', ops.ann_search.milvus(
                uri=pipe_config.milvus_uri, limit=pipe_config.limit, output_fields=pipe_config.output_fields, metric_type=pipe_config.metric_type)
            )
            .map('res', ('retrieved_urls', 'score'), lambda x: ([i.path for i in x], [i.score for i in x]))
            .flat_map(('retrieved_urls','score'), 'candidates', ops.video_copy_detection.select_video(
                top_k=pipe_config.top_k, reduce_function=pipe_config.reduce_function, reverse=pipe_config.reverse)
            )
            .map('candidates', 'retrieved_emb', FromLevelDB(path=pipe_config.leveldb_path))
            .map(('video_emb', 'retrieved_emb'), ('range', 'range_score'), ops.video_copy_detection.temporal_network(pipe_config.min_length))
            .output('url', 'candidates', 'range', 'range_score')
    )
    return p
