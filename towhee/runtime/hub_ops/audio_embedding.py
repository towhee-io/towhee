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


class AudioEmbedding:
    """
    `Audio embedding <https://towhee.io/tasks/detail/operator?field_name=Audio&task_name=Audio-Embedding>`_
    is a task that encodes audio's semantics into a set of real vectors. It is a fundamental task type that can be used in a
    variety of applications, including but not limited to reverse audio search and audio deduplication.
    """

    nnfp: HubOp = HubOp('audio_embedding.nnfp')
    """
    The audio embedding operator converts an input audio into a dense vector which can be used
    to represent the audio clip's semantics. Each vector represents for an audio clip with a fixed length
    of around 1s. This operator generates audio embeddings with fingerprinting method introduced by
    `Neural Audio Fingerprint <https://arxiv.org/abs/2010.11910>`_. The model is implemented in Pytorch.
    We've also trained the nnfp model with `FMA <https://github.com/mdeff/fma>`_ dataset (& some noise audio)
    and shared weights in this operator. The nnfp operator is suitable for audio fingerprinting.

    __init__(self, model_name: str = 'nnfp_default', model_path: str = None, framework: str = 'pytorch', device: str = None)
        model_name(`str`):
            Model name to create nnfp model with different parameters, available model names: "nnfp_default", "nnfp_hop25", "nnfp_distill".
        model_path(`str`):
            The path to model. If None, it will load default model weights.
        framework(`str`):
            Default value is pytorch. (Legacy parameter)
        device(`str`):
            Device id: cpu/cuda:{GPUID}, if not set, will try to find an available GPU device.

    __call__(self, data: List[towhee.types.AudioFrame]) -> numpy.ndarray
        data(`List[towhee.types.AudioFrame]`):
            Input audio data is a list of towhee audio frames. Create by
            `ops.audio_decode.ffmpeg() <https://towhee.io/audio-decode/ffmpeg>`_.
            The audio input should be at least 1s.

     Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        p = (
            pipe.input('path')
                .map('path', 'frame', ops.audio_decode.ffmpeg())
                .map('frame', 'vecs', ops.audio_embedding.nnfp(device='cpu'))
                .output('path', 'vecs')
        )

        DataCollection(p('test.wav')).show()

    """

    vggish: HubOp = HubOp('audio-embedding.vggish')
    """
    `vggish <https://towhee.io/audio-embedding/vggish>`_ converts an input audio into a dense vector which can be used to represent
    the audio clip's semantics. Each vector represents for an audio clip with a fixed length of around 0.9s.
    This operator is built on top of `VGGish <https://github.com/tensorflow/models/tree/master/research/audioset/vggish>`_
    with Pytorch. The model is a `VGG <https://arxiv.org/abs/1409.1556>`_ variant pre-trained with a large scale of
    audio dataset AudioSet. As suggested, it is suitable to extract features at high level or warm up a larger model.

    __init__(self, weights_path: str = None, framework: str = 'pytorch')
        weights_path(`str`):
            The path to model weights. If None, it will load default model weights.
        framework(`str`):
            Default value is pytorch. (Legacy parameter)

    __call__(self, data: List[towhee.types.AudioFrame]) -> numpy.ndarray
        data(`List[towhee.types.AudioFrame]`):
            Input audio data is a list of towhee audio frames. Create by
            `ops.audio_decode.ffmpeg() <https://towhee.io/audio-decode/ffmpeg>`_.
            The input data should represent for an audio longer than 0.9s.

    Example:

    .. code-block:: python

        from towhee import pipe, ops

        p = (
            pipe.input('path')
                .map('path', 'frame', ops.audio_decode.ffmpeg())
                .map('frame', 'vecs', ops.audio_embedding.vggish())
                .output('vecs')
        )

        p('test.wav').get()[0]

    """

    clmr: HubOp = HubOp('audio_embedding.clmr')
    """
    `clmr <https://towhee.io/audio-embedding/clmr>`_ converts an input audio into a dense vector which can be used
    to represent the audio clip's semantics. Each vector represents for an audio clip with a
    fixed length of around 2s. This operator is built on top of the original implementation of
    `CLMR <https://github.com/Spijkervet/CLMR>`_.  The default model weight provided is pretrained on
    `Magnatagatune Dataset <https://paperswithcode.com/dataset/magnatagatune>`_ with SampleCNN.

    __init__(self, framework="pytorch"):
        framework(`str`):
            Default value is pytorch. (Legacy parameter)

    __call__(self, data: List[towhee.types.AudioFrame]) -> numpy.ndarray
        data(`List[towhee.types.AudioFrame]`):
            Input audio data is a list of towhee audio frames. Create by
            `ops.audio_decode.ffmpeg() <https://towhee.io/audio-decode/ffmpeg>`_.
            The input data should represent for an audio longer than 3s.

        Returns:
            Audio embeddings in shape (num_clips, 512). Each embedding stands for features of an audio clip with length of 2.7s.

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        p = (
            pipe.input('path')
                .map('path', 'frame', ops.audio_decode.ffmpeg())
                .map('frame', 'vecs', ops.audio_embedding.clmr())
                .output('path', 'vecs')
        )

        DataCollection(p('./test.wav')).show()

    """

    data2vec: HubOp = HubOp('audio_embedding.data2vec')
    """
    `data2vec <https://towhee.io/audio-embedding/data2vec>`_ extracts features for audio with data2vec.
    The core idea is to predict latent representations of the full input data based on a masked view of the
    input in a self-distillation setup using a standard Transformer architecture.

    __init__(self, model_name = "facebook/data2vec-audio-base-960h")
        model_name(`str`):
            Default value is "facebook/data2vec-audio-base-960h". Available models: facebook/data2vec-audio-base-960h,
            facebook/data2vec-audio-large-960h, facebook/data2vec-audio-base, facebook/data2vec-audio-base-100h,
            facebook/data2vec-audio-base-10m, facebook/data2vec-audio-large, facebook/data2vec-audio-large-100h,
            facebook/data2vec-audio-large-10m

    __call__(self, data: List[towhee.types.AudioFrame]) -> numpy.ndarray
        data(`List[towhee.types.AudioFrame]`):
            Input audio data is a list of towhee audio frames. Create by
            `ops.audio_decode.ffmpeg() <https://towhee.io/audio-decode/ffmpeg>`_.

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        p = (
            pipe.input('path')
                .map('path', 'frame', ops.audio_decode.ffmpeg(sample_rate=16000))
                .map('frame', 'vecs', ops.audio_embedding.data2vec(model_name='facebook/data2vec-audio-base-960h'))
                .output('path', 'vecs')
        )

        DataCollection(p('test.wav')).show()
    """

    def __call__(self, *args, **kwargs):
        return HubOp('towhee.audio_embedding')(*args, **kwargs)
