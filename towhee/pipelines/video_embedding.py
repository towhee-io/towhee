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
        # emb op
        self.model = 'isc'

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


def _video_embedding(decode_op, emb_op, milvus_op, kv_op, norm_op, allow_triton=False, device=-1):
    op_config = {}
    if allow_triton:
        if device >= 0:
            op_config = AutoConfig.TritonGPUConfig(device_ids=[device], max_batch_size=128)
        else:
            op_config = AutoConfig.TritonCPUConfig()

    def _norm():
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

    def _unnorm():
        return (
            pipe.input('url')
                .map('url', 'id', lambda x: x)
                .flat_map('url', 'frames', decode_op)
                .map('frames', 'emb', emb_op, config=op_config)
                .map(('id', 'emb'), 'milvus_res', milvus_op)
                .window_all('emb', 'video_emb', merge_ndarray)
                .map(('id', 'video_emb'), ('insert_status'), kv_op)
                .output()
        )

    return _unnorm() if not norm_op else _norm()


def _get_embedding_op(config):
    if config.device == -1:
        device = 'cpu'
    else:
        device = config.device

    model_list = [
        'isc',
        'gmixer_24_224',
        'resmlp_24_224',
        'resmlp_12_distilled_224',
        'resmlp_12_224',
        'coat_lite_mini',
        'deit_small_patch16_224',
        'resmlp_36_224',
        'pit_xs_224',
        'convit_small',
        'resmlp_24_distilled_224',
        'tnt_s_patch16_224',
        'pit_ti_224',
        'resmlp_36_distilled_224',
        'twins_svt_small',
        'convit_tiny',
        'coat_lite_small',
        'coat_lite_tiny',
        'deit_tiny_patch16_224',
        'coat_mini',
        'gmlp_s16_224',
        'cait_xxs24_224',
        'cait_s24_224',
        'levit_128',
        'coat_tiny',
        'cait_xxs36_224',
        'levit_192',
        'levit_256',
        'levit_128s',
        'vit_small_patch32_224',
        'vit_small_patch32_384',
        'vit_small_r26_s32_224',
        'vit_small_patch16_224'
    ]
    if config.model == 'isc':
        return True, ops.image_embedding.isc(device=device)
    elif config.model in model_list:
        return True, ops.image_embedding.timm(model_name=config.model, device=device)
    raise RuntimeError(f'Unkown model: {config.model}, only support models in {model_list}.')


@AutoPipes.register
def video_embedding(config):
    """
    Define pipeline
    """
    allow_triton, emb_op = _get_embedding_op(config)
    decode_op = ops.video_decode.ffmpeg(sample_type='time_step_sample', args={'time_step': 1})
    milvus_op = ops.ann_insert.milvus_client(host=config.milvus_host, port=config.milvus_port, collection_name=config.collection)
    if config.hbase_table:
        kv_op = ops.kvstorage.insert_hbase(host=config.hbase_host, port=config.hbase_port, table=config.hbase_table)
    if config.leveldb_path:
        kv_op = ops.kvstorage.insert_leveldb(path=config.leveldb_path)
    norm_op = None if config.model == 'isc' else normalize

    return _video_embedding(decode_op, emb_op, milvus_op, kv_op, norm_op, allow_triton, config.device)
