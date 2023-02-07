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


from towhee.dc2 import ops, pipe, AutoPipes, AutoConfig


@AutoConfig.register
class SentenceSimilarityConfig:
    """
    Config of pipeline
    """
    def __init__(self):
        self.model = 'all-MiniLM-L6-v2'
        self.openai_api_key = None
        self.customize_embedding_op = None
        self.normalize_vec = True
        self.device = -1


_hf_models = ops.sentence_embedding.transformers().get_op().supported_model_names()
_openai_models = ['text-embedding-ada-002', 'text-similarity-davinci-001',
                  'text-similarity-curie-001', 'text-similarity-babbage-001',
                  'text-similarity-ada-001']


def _get_embedding_op(config):
    if config.device == -1:
        device = 'cpu'
    else:
        device = config.device

    if config.customize_embedding_op is not None:
        return True, config.customize_embedding_op

    if config.model in _hf_models:
        return True, ops.sentence_embedding.transformers(model_name=config.model,
                                                         device=device)
    if config.model in _openai_models:
        return False, ops.text_embedding.openai(engine=config.model,
                                                api_key=config.openai_api_key)
    raise RuntimeError('Unkown model: [%s], only support: %s' % (config.model, _hf_models + _openai_models))


def normalize(vec):
    import numpy as np  # pylint: disable=import-outside-toplevel
    return vec / np.linalg.norm(vec)


@AutoPipes.register
def sentence_embedding(config=None):
    """
    Define pipeline
    """
    if config is None:
        config = SentenceSimilarityConfig()

    allow_triton, emb_op = _get_embedding_op(config)
    op_config = {}
    if allow_triton:
        if config.device >= 0:
            op_config = AutoConfig.TritonGPUConfig(device_ids=[config.device], max_batch_size=128)
        else:
            op_config = AutoConfig.TritonCPUConfig()
    p = (
        pipe.input('text')
        .map('text', 'vec', emb_op, config=op_config)
    )

    if config.normalize_vec:
        p = p.map('vec', 'vec', normalize)
    return p.output('vec')
