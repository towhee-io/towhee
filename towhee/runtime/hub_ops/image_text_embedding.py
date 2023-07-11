# Copyright 2023 Zilliz. All rights reserved.
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

from towhee.runtime.factory import HubOp


class ImageTextEmbedding:
    """
    `image_text_embedding <https://towhee.io/tasks/detail/operator?field_name=Multimodal&task_name=Image/Text-Embedding>`_
    is a task that attempts to comprehend images and texts, and encode both image's and text's
    semantics into a same embedding space. It is a fundamental task type that can be used in a variety
    of applications, such as cross-modal retrieval.
    """

    clip: HubOp = HubOp('image_text_embedding.clip')
    """
    This operator extracts features for image or text with CLIP which can generate embeddings for
    text and image by jointly training an image encoder and text encoder to maximize the cosine similarity.

    __init__(self, model_name: str, modality: str, device: str = 'cpu', checkpoint_path: str = None)
        model_name(`str`):
            The model name of CLIP. Available model names: clip_vit_base_patch16, clip_vit_base_patch32
            clip_vit_large_patch14, clip_vit_large_patch14_336
        modality(`str`):
            Which modality(image or text) is used to generate the embedding.
        device(`str`):
            Device id: cpu/cuda:{GPUID}, default is cpu.
        checkpoint_path(`str`):
            The path to local checkpoint, defaults to None. If None, the operator will download and load pretrained
            model by model_name from Huggingface transformers.

    __call__(self, data: List[Union[str, towhee.type.Image]]) -> Union[List[ndarray], ndarray]:
        data(`List[Union[str, towhee.type.Image]]`)
            The data (image or text based on specified modality) to generate embedding.

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        img_pipe = (
            pipe.input('url')
            .map('url', 'img', ops.image_decode.cv2('rgb'))
            .map('img', 'vec', ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='image'))
            .output('img', 'vec')
        )

        text_pipe = (
            pipe.input('text')
            .map('text', 'vec', ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='text'))
            .output('text', 'vec')
        )

        DataCollection(img_pipe('./teddy.jpg')).show()
        DataCollection(text_pipe('A teddybear on a skateboard in Times Square.')).show()

    """

    taiyi: HubOp = HubOp('image_text_embedding.taiyi')
    """
    Chinese clip: `taiyi <https://towhee.io/image-text-embedding/taiyi>`_ extracts features for image or text,
    which can generate embeddings for text and image by jointly training an image
    encoder and text encoder to maximize the cosine similarity. This method is developed by
    `IDEA-CCNL <https://github.com/IDEA-CCNL/Fengshenbang-LM/>`_.

    __init__(self, model_name: str, modality: str, clip_checkpoint_path: str=None, text_checkpoint_path: str=None, device: str=None)
        model_name(`str`):
            The model name of Taiyi. Available model names: taiyi-clip-roberta-102m-chinese, taiyi-clip-roberta-large-326m-chinese
        modality(`str`):
            Which modality(image or text) is used to generate the embedding.
        clip_checkpoint_path(`str`):
            The weight path to load for the clip branch.
        text_checkpoint_path(`str`):
            The weight path to load for the text branch.
        device(`str`):
            The device in string, defaults to None. If None, it will enable "cuda" automatically when cuda is available.

    __call__(self, data: List[Union[str, towhee.type.Image]]) -> Union[List[ndarray], ndarray]:
        data(`List[Union[str, towhee.type.Image]]`)
            The data (image or text based on specified modality) to generate embedding.

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        img_pipe = (
            pipe.input('url')
            .map('url', 'img', ops.image_decode.cv2_rgb())
            .map('img', 'vec', ops.image_text_embedding.taiyi(model_name='taiyi-clip-roberta-102m-chinese', modality='image'))
            .output('img', 'vec')
        )

        text_pipe = (
            pipe.input('text')
            .map('text', 'vec', ops.image_text_embedding.taiyi(model_name='taiyi-clip-roberta-102m-chinese', modality='text'))
            .output('text', 'vec')
        )

        DataCollection(img_pipe('./dog.jpg')).show()
        DataCollection(text_pipe('一只小狗')).show()

    """

    bilp: HubOp = HubOp('image_text_embedding.blip')
    """
    This operator extracts features for image or text with `BLIP <https://arxiv.org/abs/2201.12086>`_
    which can generate embeddings  for text and image by jointly training an image encoder
    and text encoder to maximize the cosine similarity. This is a adaptation from
    `salesforce/BLIP <https://github.com/salesforce/BLIP>`_.

    __init__(self, model_name: str, modality: str, device:str = 'cpu', checkpoint_path: str = None)
        model_name(`str`):
            The model name of CLIP. Available model names: blip_itm_base_coco, blip_itm_large_coco,
            blip_itm_base_flickr, blip_itm_large_flickr
        modality(`str`):
            Which modality(image or text) is used to generate the embedding.
        device(`str`):
            Device id: cpu/cuda:{GPUID}, default is cpu.
        checkpoint_path(`str`):
            The path to local checkpoint, defaults to None. If None, the operator will download and load pretrained
            model by model_name from Huggingface transformers.

    __call__(self, data: List[Union[str, towhee.type.Image]]) -> Union[List[ndarray], ndarray]:
        data(`List[Union[str, towhee.type.Image]]`)
            The data (image or text based on specified modality) to generate embedding.

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        img_pipe = (
            pipe.input('url')
            .map('url', 'img', ops.image_decode.cv2('rgb'))
            .map('img', 'vec', ops.image_text_embedding.blip(model_name='blip_itm_base_coco', modality='image'))
            .output('img', 'vec')
        )

        text_pipe = (
            pipe.input('text')
            .map('text', 'vec', ops.image_text_embedding.blip(model_name='blip_itm_base_coco', modality='text'))
            .output('text', 'vec')
        )

        DataCollection(img_pipe('./teddy.jpg')).show()
        DataCollection(text_pipe('A teddybear on a skateboard in Times Square.')).show()

    """

    def __call__(self, *args, **kwargs):
        """
        Resolve the conflict issue that may be caused by ops users omitting the towhee namespace during use.
        """
        return HubOp('towhee.image_text_embedding')(*args, **kwargs)

