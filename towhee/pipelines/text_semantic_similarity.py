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


from towhee.dc2 import ops, pipe
from towhee.runtime import AutoPipes, AutoConfig


@AutoConfig.register
class TextEmbeddingConfig:
    """
    Config of pipeline
    """
    def __init__(self):
        self.model_provider = 'hf'  # 'hf' | 'openai' | 'sbert'
        self.model = 'all-MiniLM-L12-v2'

        self.openai_api_key = None

        self.insert_to = None  # None | milvus
        self.host= '127.0.0.1'
        self.port= '19530'
        self.collection_name = None

        self.device = -1


def _text_semantic_similarity(embedding_op, milvus_op=None, allow_triton=False, device=-1):
    op_config = {}
    if allow_triton:
        if device >= 0:
            op_config = AutoConfig.TritonGPUConfig(device_ids=[device], max_batch_size=128)
        else:
            op_config = AutoConfig.TritonCPUConfig()

    def _only_embedding():
        return (
            pipe.input('text')
            .map('text', 'vec', embedding_op, config=op_config)
            .output('vec')
        )

    def _to_milvus():
        return (
            pipe.input('text')
            .map('text', 'vec', embedding_op, config=op_config)
            .map(('text', 'vec'), (), milvus_op)
            .output()
        )

    if milvus_op is None:
        return _only_embedding()
    return _to_milvus()


def _get_embedding_op(config):
    if config.device == -1:
        device = 'cpu'
    else:
        device = config.device

    if config.model_provider == 'hf':
        return True, ops.text_embedding.transformers(model_name=config.model,
                                                     device=device)
    if config.model_provider == 'openai':
        return False, ops.text_embedding.openai(engine=config.model,
                                                api_key=config.openai_api_key)
    if config.model_provider == 'sbert':
        return True, ops.text_embedding.sentence_transformers(model_name=config.model,
                                                              device=device)
    raise RuntimeError('Unkown model provider:%s, only support hf | openai | sbert' % (config.model_provider))


@AutoPipes.register
def text_semantic_similarity(config):
    """
    Define pipeline
    """
    allow_triton, emb_op = _get_embedding_op(config)
    milvus_op = None
    if config.insert_to is not None:
        milvus_op = ops.ann_insert.milvus_client(host=config.host,
                                                 port=config.port,
                                                 collection_name=config.collection_name)
    return _text_semantic_similarity(emb_op, milvus_op, allow_triton, config.device)
