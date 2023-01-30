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
class VideoEmbeddingConfig:
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

        # kv op
        self.hbase_host = '127.0.0.1'
        self.hbase_port = 9090
        self.hbase_table = None
        self.leveldb_path = None

        self.device = -1


def _video_embedding(decode_op, emb_op, milvus_op, kv_op, allow_triton=False, device=-1):
    op_config = {}
    if allow_triton:
        if device >= 0:
            op_config = AutoConfig.TritonGPUConfig(device_ids=[device], max_batch_size=128)
        else:
            op_config = AutoConfig.TritonCPUConfig()

    def _insert():
        return (
            pipe.input('url')
                .map('url', 'id', lambda x: x)
                .flat_map('url', 'frames', decode_op)
                .map('frames', 'emb', emb_op, config=op_config)
                .map('emb', 'emb', normalize)
                .map(('id', 'emb'), 'milvus_res', milvus_op)
                .window_all('emb', 'video_emb', merge_ndarray)
                .map(('id', 'video_emb'), ('insert_status'), kv_op)
                .output()
        )

    return _insert()


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
def video_embedding(config):
    """
    Define pipeline
    """
    allow_triton, emb_op = _get_embedding_op(config)
    decode_op = ops.video_decode.ffmpeg(sample_type=config.sample_type, args=config.deocde_args)
    milvus_op = ops.ann_insert.milvus(host=config.milvus_host, port=config.milvus_port, collection=config.collection)
    if config.hbase_table:
        kv_op = ops.kvstorage.insert_hbase(host=config.hbase_host, port=config.hbase_port, table=config.hbase_table)
    if config.leveldb_path:
        kv_op = ops.kvstorage.insert_leveldb(config.leveldb_path)

    return _video_embedding(decode_op, emb_op, milvus_op, kv_op, allow_triton, config.device)
