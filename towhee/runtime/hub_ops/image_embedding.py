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


class ImageEmbedding:
    """
     `image_embedding <https://towhee.io/tasks/detail/operator?field_name=Computer-Vision&task_name=Image-Embedding>`_
      is a task that attempts to comprehend an entire image as a whole
      and encode the image's semantics into a real vector.
      It is a fundamental task type that can be used in a variety of applications,
      including but not limited to reverse image search and image deduplication.
    """

    timm: HubOp = HubOp('image_embedding.timm')
    """
    `timm <https://towhee.io/image-embedding/timm>`_ operator extracts features for image
    with pre-trained models provided by `Timm <https://github.com/huggingface/pytorch-image-models>`_.
    Visual models like VGG, Resnet, ViT and so on are all available here.

    __init__(self, model_name: str = None, num_classes: int = 1000, skip_preprocess: bool = False, device: str = None, checkpoint_path: str = None)
      model_name(`str`):
         The model name in string. The default value is resnet34.
      num_classes(`int`):
         The number of classes. The default value is 1000.
      skip_preprocess(`bool`):
          Whether to skip image pre-process, Default value is False.
      device (`str`):
         Device id: cpu/cuda:{GPUID}, if not set, will try to find an available GPU device.
      checkpoint_path(`str`):
         Local weights path, if not set, will download remote weights.

    __call__(self, data: Union[List['towhee.types.Image'], 'towhee.types.Image']) ->  Union[List[ndarray], ndarray]:
      data('towhee.types.Image'):
         decode by ops.image_decode.cv2 or ops.image_decode.nvjpeg


    Examples:

    .. code-block:: python

      from towhee import pipe, ops, DataCollection

      p = (
         pipe.input('path')
               .map('path', 'img', ops.image_decode())
               .map('img', 'vec', ops.image_embedding.timm(model_name='resnet50'))
               .output('img', 'vec')
      )

      DataCollection(p('towhee.jpeg')).show()
    """

    isc: HubOp = HubOp('image_embedding.isc')
    """
    `isc <https://towhee.io/image-embedding/isc>`_ operator extracts features for image top ranked models from
    `Image Similarity Challenge 2021 <https://github.com/facebookresearch/isc2021>`_- Descriptor Track.
    The default pretrained model weights are from The 1st Place Solution of ISC21
    `Descriptor Track <https://github.com/lyakaap/ISC21-Descriptor-Track-1st>`_.

    __init__(self, timm_backbone: str = 'tf_efficientnetv2_m_in21ft1k', img_size: int = 512,
             checkpoint_path: str = None,
             skip_preprocess: bool = False,
             device: str = None)
      timm_backbone(`str`):
         backbone of model, the default value is tf_efficientnetv2_m_in21ft1k.
      img_size(`int`):
         Resize image, default is 512.
      checkpoint_path(`str`):
         If not set, will download remote weights.
      skip_preprocess(`bool`):
          Whether to skip image pre-process, Default value is False.
      device (`str`):
         Device id: cpu/cuda:{GPUID}, if not set, will try to find an available GPU device.

    __call__(self, data: Union[List['towhee.types.Image'], 'towhee.types.Image']) ->  Union[List[ndarray], ndarray]
      data('towhee.types.Image'):
         decode by ops.image_decode.cv2 or ops.image_decode.nvjpeg
    """

    data2vec: HubOp = HubOp('image_embedding.data2vec')
    """
    `data2vec <https://towhee.io/image-embedding/data2vec>`_ operator extracts features for
    image with `data2vec <https://arxiv.org/abs/2202.03555>`_.
    The core idea is to predict latent representations of the full input data based
    on a masked view of the input in a self-distillation setup using a standard Transformer architecture.

    __init__(self, model_name='facebook/data2vec-vision-base')
      model_name(`str`):
         The model name in string. The default value is "facebook/data2vec-vision-base-ft1k".
         Supported model name:
            facebook/data2vec-vision-base-ft1k

            facebook/data2vec-vision-large-ft1k

    __call__(self, img: 'towhee.types.Image') -> numpy.ndarray:
      data('towhee.types.Image'):
         decode by ops.image_decode.cv2 or ops.image_decode.nvjpeg

    Example:

    .. code-block:: python

      from towhee import pipe, ops, DataCollection

      p = (
         pipe.input('path')
         .map('path', 'img', ops.image_decode())
         map('img', 'vec', ops.image_embedding.data2vec(model_name='facebook/data2vec-vision-base-ft1k'))
         .output('img', 'vec')
      )

      DataCollection(p('towhee.jpeg')).show()
    """

    mpvit: HubOp = HubOp('image_embedding.mpvit')
    """
    `mpvit <https://towhee.io/image-embedding/mpvit>`_ extracts features for images with
    Multi-Path Vision Transformer (MPViT) which can generate embeddings for images.
    MPViT embeds features of the same size~(i.e., sequence length) with patches of different
    scales simultaneously by using overlapping convolutional patch embedding.
    Tokens of different scales are then independently fed into the Transformer encoders via
    multiple paths and the resulting features are aggregated, enabling both fine and coarse
    feature representations at the same feature level.

    __init__(self, model_name, num_classes: int = 1000, weights_path: str = None, device: str = None, skip_preprocess: bool = False)
      model_name(`str`):
        Pretrained model name include mpvit_tiny, mpvit_xsmall, mpvit_small or mpvit_base,
        all of which are pretrained on ImageNet-1K dataset,
        for more information, please refer the original `MPViT github page <https://github.com/youngwanLEE/MPViT>`_.
      weights_path(`str`):
         Your local weights path, default is None, which means using the pretrained model weights.
      device (`str`):
         Device id: cpu/cuda:{GPUID}, if not set, will try to find an available GPU device.
      num_classes(`int`):
         The number of classes. The default value is 1000.
      skip_preprocess(`bool`):
          Whether to skip image pre-process, Default value is False.

    __call__(self, data: Union[List['towhee.types.Image'], 'towhee.types.Image']) -> Union[List[ndarray], ndarray]
      data('towhee.types.Image'):
         decode by ops.image_decode.cv2 or ops.image_decode.nvjpeg

    Example:

    .. code-block:: python

      from towhee import pipe, ops, DataCollection

      p = (
         pipe.input('path')
            .map('path', 'img', ops.image_decode())
            .map('img', 'vec', ops.image_embedding.mpvit(model_name='mpvit_base'))
            .output('img', 'vec')
      )

      DataCollection(p('towhee.jpeg')).show()
    """

    swag: HubOp = HubOp('image_embedding.swag')
    """
    `swag <https://towhee.io/image-embedding/swag>`_ extracts features for image with pretrained SWAG models from Torch Hub.
    `SWAG <https://github.com/facebookresearch/SWAG>`_ implements models from the paper Revisiting Weakly Supervised Pre-Training
    of Visual Perception Models. To achieve higher accuracy in image classification,
    SWAG uses hashtags to perform weakly supervised learning instead of fully supervised pretraining with image class labels.

    __init__(self, model_name: str, skip_preprocess: bool = False)
      model_name(`str`):
          The model name in string. The default value is "vit_b16_in1k". Supported model names:
          vit_b16_in1k, vit_l16_in1k, vit_h14_in1k, regnety_16gf_in1k, regnety_32gf_in1k,  regnety_128gf_in1k
      skip_preprocess(`bool`):
          Whether to skip image pre-process, Default value is False.

    __call__(self, img: 'towhee.types.Image') -> ndarray:
      data('towhee.types.Image'):
         decode by ops.image_decode.cv2 or ops.image_decode.nvjpeg

    Example:

    .. code-block:: python

      from towhee import pipe, ops, DataCollection

      p = (
         pipe.input('path')
            .map('path', 'img', ops.image_decode())
            .map('img', 'vec', ops.image_embedding.swag(model_name='vit_b16_in1k'))
            .output('img', 'vec')
      )

      DataCollection(p('towhee.jpeg')).show()
    """

    def __call__(self, *args, **kwargs):
        """
        Resolve the conflict issue that may be caused by ops users omitting the towhee namespace during use.
        """
        return HubOp('towhee.image_embedding')(*args, **kwargs)
