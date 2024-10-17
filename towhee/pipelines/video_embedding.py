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

from typing import Optional
from pydantic import BaseModel

from towhee import ops, pipe, AutoPipes, AutoConfig


def merge_ndarray(x):
    import numpy as np
    return np.concatenate(x).reshape(-1, x[0].shape[0])


@AutoConfig.register
class VideoEmbeddingConfig(BaseModel):
    """
    Config of pipeline
    """
    # decode op
    # The range in the video to deocode
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # emb op
    model: Optional[str] = 'isc'
    img_size: Optional[int] = 512

    # milvus op
    milvus_host: Optional[str] = '127.0.0.1'
    milvus_port: Optional[str] = '19530'
    collection: Optional[str] = None

    # kv op
    hbase_host: Optional[str] = '127.0.0.1'
    hbase_port: Optional[int] = 9090
    hbase_table: Optional[str] = None
    leveldb_path: Optional[str] = None

    device: Optional[int] = -1


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
                .flat_map('url', 'frames', decode_op)
                .map('frames', 'emb', emb_op, config=op_config)
                .map('emb', 'emb', norm_op)
                .map(('url', 'emb'), 'milvus_res', milvus_op)
                .window_all('emb', 'video_emb', merge_ndarray)
                .map(('url', 'video_emb'), ('insert_status'), kv_op)
                .output()
        )

    def _unnorm():
        return (
            pipe.input('url')
                .flat_map('url', 'frames', decode_op)
                .map('frames', 'emb', emb_op, config=op_config)
                .map(('url', 'emb'), 'milvus_res', milvus_op)
                .window_all('emb', 'video_emb', merge_ndarray)
                .map(('url', 'video_emb'), ('insert_status'), kv_op)
                .output()
        )

    if norm_op:
        return _norm()
    return _unnorm()

def _get_embedding_op(config):
    if config.device == -1:
        device = 'cpu'
    else:
        device = config.device

    model_list = [
        'isc',
        'gmixer_24_224',
        'resmlp_12_224',
        'coat_lite_mini',
        'deit_small_patch16_224',
        'pit_xs_224',
        'convit_small',
        'tnt_s_patch16_224',
        'pit_ti_224',
        'resmlp_36_distilled_224',
        'convit_tiny',
        'coat_lite_small',
        'coat_lite_tiny',
        'deit_tiny_patch16_224',
        'cait_xxs24_224',
        'cait_s24_224',
        'cait_xxs36_224',
        'vit_small_patch32_224',
        'vit_small_patch32_384',
        'vit_small_r26_s32_224',
        'vit_small_patch16_224'
    ]
    if config.model == 'isc':
        return True, ops.image_embedding.isc(img_size=config.img_size, device=device)
    elif config.model in model_list:
        return True, ops.image_embedding.timm(model_name=config.model, device=device)
    raise RuntimeError(f'Unknown model: {config.model}, only support models in {model_list}.')


@AutoPipes.register
def video_embedding(config):
    """
    Define pipeline
    """
    if config.start_time and config.end_time and config.start_time >= config.end_time:
        raise ValueError('Start time should be earlier than end time in pipeline config.')
    allow_triton, emb_op = _get_embedding_op(config)
    decode_op = ops.video_decode.ffmpeg(
        start_time=config.start_time, end_time=config.end_time, sample_type='time_step_sample', args={'time_step': 1}
    )
    milvus_op = ops.ann_insert.milvus_client(host=config.milvus_host, port=config.milvus_port, collection_name=config.collection)
    if config.hbase_table:
        kv_op = ops.kvstorage.insert_hbase(host=config.hbase_host, port=config.hbase_port, table=config.hbase_table)
    if config.leveldb_path:
        kv_op = ops.kvstorage.insert_leveldb(path=config.leveldb_path)
    norm_op = None if config.model == 'isc' else ops.towhee.np_normalize()

    return _video_embedding(decode_op, emb_op, milvus_op, kv_op, norm_op, allow_triton, config.device)
