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


from towhee.dc2 import pipe, ops, AutoPipes, AutoConfig


@AutoConfig.register
class TextImageEmbeddingConfig:
    def __init__(self):
        self.model = 'clip_vit_base_patch16'
        self.modality = 'image'
        self.normalize_vec = True
        self.customize_embedding_op = None
        self.device = -1


def _get_embedding_op(config):
    if config.device == -1:
        device = 'cpu'
    else:
        device = config.device

    if config.customize_embedding_op is not None:
        return config.customize_embedding_op

    return ops.image_text_embedding.clip(model_name=config.model, modality=config.modality, device=device)


def _image_embedding(emb_op, op_config):
    return (
        pipe.input('url')
        .map('url', 'image', ops.image_decode.cv2_rgb())
        .map('image', 'vec', emb_op, config=op_config)
    )


def _text_embedding(emb_op, op_config):
    return (
        pipe.input('text')
        .map('text', 'vec', emb_op, op_config)
    )


@AutoPipes.register
def text_image_embedding(config=None):

    if config is None:
        config = TextImageEmbeddingConfig()

    emb_op = _get_embedding_op(config)

    if config.device >= 0:
        op_config = AutoConfig.TritonGPUConfig(device_ids=[config.device], max_batch_size=128)
    else:
        op_config = AutoConfig.TritonCPUConfig()

    if config.modality == 'image':
        p = _image_embedding(emb_op, op_config)
    elif config.modality == 'text':
        p = _text_embedding(emb_op, op_config)
    else:
        raise RuntimeError('Unknown modality: %s, please use image | text' % config.modality)

    if config.normalize_vec:
        p = p.map('vec', 'vec', ops.towhee.np_normalize())
    return p.output('vec')
