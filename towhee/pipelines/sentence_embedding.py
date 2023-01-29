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
        self.model = 'all-MiniLM-L12-v2'
        self.openai_api_key = None
        self.customize_embedding_op = None
        self.device = -1


_hf_models = ['sentence-t5-xxl', 'sentence-t5-xl', 'sentence-t5-large', 'paraphrase-mpnet-base-v2',
              'gtr-t5-xxl', 'gtr-t5-large', 'gtr-t5-xl', 'paraphrase-multilingual-mpnet-base-v2',
              'paraphrase-distilroberta-base-v2', 'all-mpnet-base-v1', 'all-roberta-large-v1',
              'all-mpnet-base-v2', 'all-MiniLM-L12-v2', 'all-distilroberta-v1', 'all-MiniLM-L12-v1',
              'gtr-t5-base', 'paraphrase-multilingual-MiniLM-L12-v2', 'paraphrase-MiniLM-L12-v2',
              'all-MiniLM-L6-v1', 'paraphrase-TinyBERT-L6-v2', 'all-MiniLM-L6-v2', 'paraphrase-albert-small-v2',
              'multi-qa-mpnet-base-cos-v1', 'paraphrase-MiniLM-L3-v2', 'multi-qa-distilbert-cos-v1',
              'multi-qa-mpnet-base-dot-v1', 'msmarco-distilbert-base-v4', 'msmarco-distilbert-base-tas-b',
              'distiluse-base-multilingual-cased-v2', 'multi-qa-MiniLM-L6-cos-v1', 'multi-qa-distilbert-dot-v1',
              'distiluse-base-multilingual-cased-v1', 'msmarco-bert-base-dot-v5', 'paraphrase-MiniLM-L6-v2', 'multi-qa-MiniLM-L6-dot-v1',
              'msmarco-distilbert-dot-v5', 'bert-base-nli-mean-tokens', 'bert-large-uncased-whole-word-masking',
              'average_word_embeddings_komninos', 'realm-cc-news-pretrained-embedder', 'distilbert-base-uncased',
              'average_word_embeddings_glove.6B.300d', 'dpr-ctx_encoder-multiset-base', 'dpr-ctx_encoder-single-nq-base',
              'microsoft/deberta-xlarge', 'facebook/bart-large', 'bert-base-uncased', 'microsoft/deberta-xlarge-mnli',
              'gpt2-xl', 'bert-large-uncased']
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
    return (
        pipe.input('text')
        .map('text', 'vec', emb_op, config=op_config)
        .output('vec')
    )
