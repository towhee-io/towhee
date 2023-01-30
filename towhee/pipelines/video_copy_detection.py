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


def merge_ndarray(x):
    import numpy as np
    return np.concatenate(x).reshape(-1, x[0].shape[0])


def normalize(x):
    import numpy as np
    return x / np.linalg.norm(x, axis=0)


@AutoConfig.register
class VideoCopyDetectionConfig:
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
        self.milvus_host = '127.0.0.1'
        self.milvus_port = '19530'
        self.collection = None
        self.limit = 2
        self.metric_type = 'IP'

        # kv op
        self.hbase_host = '127.0.0.1'
        self.hbase_port = 9090
        self.hbase_table = None
        self.leveldb_path = None

        # select video op
        self.top_k = 2

        # tn op
        self.min_length = 1

        self.device = -1


def _video_copy_detection(decode_op, emb_op, milvus_op, kv_op, select_op, tn_op, allow_triton=False, device=-1):
    op_config = {}
    if allow_triton:
        if device >= 0:
            op_config = AutoConfig.TritonGPUConfig(device_ids=[device], max_batch_size=128)
        else:
            op_config = AutoConfig.TritonCPUConfig()

    def _search():
        return (
            pipe.input('url')
                .map('url', 'id', lambda x: x)
                .flat_map('url', 'frames', decode_op)
                .map('frames', 'emb', emb_op, config=op_config)
                .map('emb', 'emb', normalize)
                .map('emb', 'res', milvus_op)
                .window_all('res', 'res', lambda x:[i for y in x for i in y])
                .map('res', ('retrieved_ids', 'score'), lambda x: ([i.path for i in x], [i.score for i in x]))
                .window_all('emb', 'video_emb', merge_ndarray)
                .flat_map(('retrieved_ids','score'),'candidates', select_op)
                .map('candidates', 'retrieved_emb', kv_op)
                .map(('video_emb', 'retrieved_emb'), ('range', 'range_score'), tn_op)
                .output('id', 'candidates', 'range', 'range_score')
        )

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
def video_copy_detection(config):
    """
    Define pipeline
    """
    allow_triton, emb_op = _get_embedding_op(config)
    decode_op = ops.video_decode.ffmpeg(sample_type=config.sample_type, args=config.deocde_args)

    milvus_op = ops.ann_search.milvus(
        host=config.milvus_host,
        port=config.milvus_port,
        collection=config.collection,
        limit=config.limit,
        output_fields=['path'],
        metric_type=config.metric_type
    )
    if config.hbase_table:
        kv_op = ops.kvstorage.search_hbase(host=config.hbase_host, port=config.hbase_port, table=config.hbase_table, is_ndarray=True)
    if config.leveldb_path:
        kv_op = ops.kvstorage.from_leveldb(leveldb_path=config.leveldb_path, is_ndarray=True)
    select_op = ops.video_copy_detection.select_video(top_k=config.top_k, reduce_function='sum', reverse=True)
    tn_op = ops.video_copy_detection.temporal_network(min_length=config.min_length)

    return _video_copy_detection(decode_op, emb_op, milvus_op, kv_op, select_op, tn_op, allow_triton, config.device)
