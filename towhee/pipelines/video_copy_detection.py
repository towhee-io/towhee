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

from towhee import ops, pipe, AutoPipes, AutoConfig


def merge_ndarray(x):
    import numpy as np
    return np.concatenate(x).reshape(-1, x[0].shape[0])


class DuplicateFilter:
    """
    Duplicate Filter.
    """
    def __init__(self, threshold):
        self._threshold = threshold

    def __call__(self, segs, scores, src):
        duration = src.shape[0]
        weights = [0] * duration
        for i, j in zip(segs, scores):
            for idx in range(i[0], i[2] + 1):
                weights[idx] = max(weights[idx], j)

        return (sum(weights) / duration) > self._threshold


@AutoConfig.register
class VideoCopyDetectionConfig:
    """
    Config of pipeline
    """
    def __init__(self):
        # decode op
        # The range in the video to deocode
        self.start_time = None
        self.end_time = None

        # emb op
        self.model = 'isc'
        self.img_size = 512

        # milvus op
        self.milvus_host = '127.0.0.1'
        self.milvus_port = '19530'
        self.collection = None
        # The number of embeddings return from milvus search op
        self.milvus_search_limit = 64

        # kv op
        self.hbase_host = '127.0.0.1'
        self.hbase_port = 9090
        self.hbase_table = None
        self.leveldb_path = None

        # select video op
        # The number of nearest videos to return by the pipeline
        self.top_k = 100

        # tn op
        # The minimal similar frame(s) that the return videos should contain
        self.min_similar_length = 1

        # filter op
        self.threshold = None

        # tracer
        self.tracer = False

        self.device = -1


def _video_copy_detection(decode_op, emb_op, milvus_op, kv_op, select_op, tn_op, norm_op, filter_op, tracer, allow_triton=False, device=-1):
    op_config = {}
    if allow_triton:
        if device >= 0:
            op_config = AutoConfig.TritonGPUConfig(device_ids=[device], max_batch_size=128)
        else:
            op_config = AutoConfig.TritonCPUConfig()
    if norm_op:
        emb_pipe = (
            pipe.input('url')
                .flat_map('url', 'frames', decode_op)
                .map('frames', 'emb', emb_op, config=op_config)
                .map('emb', 'emb', norm_op)
        )
    else:
        emb_pipe = (
            pipe.input('url')
                .flat_map('url', 'frames', decode_op)
                .map('frames', 'emb', emb_op, config=op_config)
        )

    search_pipe = (
        emb_pipe.flat_map('emb', 'res', milvus_op)
            .window_all('res', ('retrieved_urls', 'score'), lambda x: ([i[2] for i in x], [i[1] for i in x]))
            .window_all('emb', 'video_emb', merge_ndarray)
            .flat_map(('retrieved_urls','score'),'candidates', select_op)
            .map('candidates', 'retrieved_emb', kv_op)
    )

    if filter_op:
        detect_pipe = (
            search_pipe.map(('video_emb', 'retrieved_emb'), ('similar_segment', 'segment_score'), tn_op)
                .filter(
                    ('candidates', 'similar_segment', 'segment_score'),
                    ('candidates', 'similar_segment', 'segment_score'),
                    ('similar_segment', 'segment_score', 'video_emb'),
                    filter_op
                )
                .output('url', 'candidates', 'similar_segment', 'segment_score', tracer=tracer)
        )
    else:
        detect_pipe = (
            search_pipe.map(('video_emb', 'retrieved_emb'), ('similar_segment', 'segment_score'), tn_op)
                .output('url', 'candidates', 'similar_segment', 'segment_score', tracer=tracer)
        )

    return detect_pipe


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
    raise RuntimeError(f'Unkown model: {config.model}, only support models in {model_list}.')


@AutoPipes.register
def video_copy_detection(config):
    """
    Define pipeline
    """
    if config.start_time and config.end_time and config.start_time >= config.end_time:
        raise ValueError('Start time should be earlier than end time in pipeline config.')
    allow_triton, emb_op = _get_embedding_op(config)
    decode_op = ops.video_decode.ffmpeg(
        start_time=config.start_time, end_time=config.end_time, sample_type='time_step_sample', args={'time_step': 1}
    )
    milvus_op = ops.ann_search.milvus_client(
        host=config.milvus_host,
        port=config.milvus_port,
        collection_name=config.collection,
        output_fields=['path'],
        metric_type='IP',
        limit = config.milvus_search_limit
    )
    if config.hbase_table:
        kv_op = ops.kvstorage.search_hbase(host=config.hbase_host, port=config.hbase_port, table=config.hbase_table, is_ndarray=True)
    if config.leveldb_path:
        kv_op = ops.kvstorage.from_leveldb(path=config.leveldb_path, is_ndarray=True)
    select_op = ops.video_copy_detection.select_video(top_k=config.top_k, reduce_function='sum', reverse=True)
    tn_op = ops.video_copy_detection.temporal_network(min_length=config.min_similar_length)
    norm_op = None if config.model == 'isc' else ops.towhee.np_normalize()
    filter_op = DuplicateFilter(config.threshold) if config.threshold else None


    return _video_copy_detection(
        decode_op, emb_op, milvus_op, kv_op, select_op, tn_op, norm_op, filter_op, config.tracer, allow_triton, config.device
    )
