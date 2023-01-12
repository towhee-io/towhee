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

from towhee.dc2 import ops, pipe, AutoPipes, AutoConfig


def get_image(x):
    from towhee.types import Image
    return Image(x.__array__(), 'RGB')


def merge_ndarray(x):
    import numpy as np
    return np.concatenate(x).reshape(-1, x[0].shape[0])


def normalize(x):
    import numpy as np
    return x / np.linalg.norm(x, axis=0)


@AutoConfig.register
class VideoCopyConfig:
    """
    Config of pipeline
    """
    def __init__(self):
        # decode op
        self.sample_type = 'time_step_sample'
        self.deocde_args = {'time_step': 1}

        # emb op
        self.model = 'resnet34'  # 'resnet34' | 'isc'

        # milvus op
        self.host = '127.0.0.1'
        self.port = '19530'
        self.collection = None
        self.limit = 2
        self.metric_type='IP'

        # leveldb op
        self.leveldb_path = 'url_vec.db'

        # select video op
        self.top_k=2

        # tn op
        self.min_length = 1

        self.device = -1


def _video_copy(insert, decode_op, emb_op, milvus_op, leveldb_op, select_op, tn_op, allow_triton=False, device=-1):
    op_config = {}
    if allow_triton:
        if device >= 0:
            op_config = AutoConfig.TritonGPUConfig(device_ids=[device], max_batch_size=128)
        else:
            op_config = AutoConfig.TritonCPUConfig()

    def _insert():
        return (
            pipe.input('url')
                .flat_map('url', 'frames', decode_op)
                .map('frames', 'img', get_image)
                .map('img', 'emb', emb_op, config=op_config)
                .map('emb', 'emb', normalize)
                .map(('url', 'emb'), 'milvus_res', milvus_op)
                .window_all('emb', 'video_emb', merge_ndarray)
                .map(('url', 'video_emb'), ('insert_status'), leveldb_op)
                .output('url', 'insert_status')
        )

    def _search():
        return (
            pipe.input('url')
                .flat_map('url', 'frames', decode_op)
                .map('frames', 'img', get_image)
                .map('img', 'emb', emb_op, config=op_config)
                .map('emb', 'emb', normalize)
                .map('emb', 'res', milvus_op)
                .window_all('res', 'res', lambda x:[i for y in x for i in y])
                .map('res', ('retrieved_urls', 'score'), lambda x: ([i.path for i in x], [i.score for i in x]))
                .window_all('emb', 'video_emb', merge_ndarray)
                .flat_map(('retrieved_urls','score'),'candidates', select_op)
                .map('candidates', 'retrieved_emb', leveldb_op)
                .map(('video_emb', 'retrieved_emb'), ('range', 'range_score'), tn_op)
                .output('url', 'candidates', 'range', 'range_score')
        )

    if insert:
        return _insert()
    return _search()


def _get_embedding_op(config):
    if config.device == -1:
        device = 'cpu'
    else:
        device = config.device

    if config.model == 'resnet34':
        return True, ops.image_embedding.timm(model_name='resnet34', device=device)
    if config.model_provider == 'isc':
        return True, ops.image_embedding.isc(device=device)
    raise RuntimeError('Unkown model provider:%s, only support resnet34 | isc' % (config.model_provider))


@AutoPipes.register
def video_copy(insert: bool, config):
    """
    Define pipeline
    """
    allow_triton, emb_op = _get_embedding_op(config)
    decode_op = ops.video_decode.ffmpeg(sample_type=config.sample_type, args=config.deocde_args)

    if insert:
        milvus_op = ops.ann_insert.milvus(host=config.host,
                                          port=config.port,
                                          collection=config.collection)
        leveldb_op = ops.kvstorage.insert_leveldb(config.leveldb_path)
        select_op = None
        tn_op = None
    else:
        milvus_op = ops.ann_search.milvus(host=config.host,
                                          port=config.port,
                                          collection=config.collection,
                                          limit=config.limit,
                                          output_fields=['path'],
                                          metric_type=config.metric_type)
        leveldb_op = ops.kvstorage.from_leveldb(config.leveldb_path, True)
        select_op = ops.video_copy_detection.select_video(top_k=config.top_k,
                                                          reduce_function='sum',
                                                          reverse=True)
        tn_op = ops.video_copy_detection.temporal_network(min_length=config.min_length)

    return _video_copy(insert, decode_op, emb_op, milvus_op, leveldb_op, select_op, tn_op, allow_triton, config.device)
