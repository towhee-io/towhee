&nbsp;

<p align="center">
    <img src="towhee_logo.png#gh-light-mode-only" width="60%"/>
    <img src="assets/towhee_logo_dark.png#gh-dark-mode-only" width="60%"/>
</p>


<h3 align="center">
  <p style="text-align: center;"> <span style="font-weight: bold; font: Arial, sans-serif;">x</span>2vec, Towhee is all you need! </p>
</h3>

<h3 align="center">
  <p style="text-align: center;">
  <a href="README.md" target="_blank">ENGLISH</a> | <a href="README_CN.md">中文文档</a>
  </p>
</h3>

<div class="column" align="middle">
  <a href="https://slack.towhee.io">
    <img src="https://img.shields.io/badge/join-slack-orange?style=flat" alt="join-slack"/>
  </a>
  <a href="https://twitter.com/towheeio">
    <img src="https://img.shields.io/badge/follow-twitter-blue?style=flat" alt="twitter"/>
  </a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img src="https://img.shields.io/badge/license-apache2.0-green?style=flat" alt="license"/>
  </a>
  <a href="https://github.com/towhee-io/towhee/actions/workflows/pylint.yml">
    <img src="https://img.shields.io/github/workflow/status/towhee-io/towhee/Workflow%20for%20pylint/main?label=pylint&style=flat" alt="github actions"/>
  </a>
  <a href="https://app.codecov.io/gh/towhee-io/towhee">
    <img src="https://img.shields.io/codecov/c/github/towhee-io/towhee?style=flat" alt="coverage"/>
  </a>
</div>

&nbsp;

[Towhee](https://towhee.io) makes it easy to build neural data processing pipelines for AI applications.
We provide hundreds of models, algorithms, and transformations that can be used as standard pipeline building blocks.
You can use Towhee's Pythonic API to build a prototype of your pipeline and
automatically optimize it for production-ready environments.

:art:&emsp;**Various Modalities:** Towhee supports data processing on a variety of modalities, including images, videos, text, audio, molecular structures, etc.

:mortar_board:&emsp;**SOTA Models:** Towhee provides SOTA models across 5 fields (CV, NLP, Multimodal, Audio, Medical), 15 tasks, and 140+ model architectures. These include BERT, CLIP, ViT, SwinTransformer, MAE, and data2vec, all pretrained and ready to use.

:package:&emsp;**Data Processing:** Towhee also provides traditional methods alongside neural network models to help you build practical data processing pipelines. We have a rich pool of operators available, such as video decoding, audio slicing, frame sampling, feature vector dimension reduction, ensembling, and database operations.

:snake:&emsp;**Pythonic API:** Towhee includes a Pythonic method-chaining API for describing custom data processing pipelines. We also support schemas, which makes processing unstructured data as easy as handling tabular data.

## What's New
**v1.0.0rc1 May. 4, 2023**
* Add trainer to operators: 
[*timm*](https://towhee.io/image-embedding/timm), [*isc*](https://towhee.io/image-embedding/isc), [*transformers*](https://towhee.io/text-embedding/transformers), [*clip*](https://towhee.io/image-text-embedding/clip)
* Add GPU video decoder: 
[*VPF*](https://towhee.io/video-decode/VPF)
* All towhee pipelines can be converted into Nvidia Triton services.


**v0.9.0 Dec. 2, 2022**
* Added one video classification model:
[*Vis4mer*](https://github.com/towhee-io/towhee/tree/branch0.9.0/towhee/models/vis4mer)
* Added three visual backbones:
[*MCProp*](https://github.com/towhee-io/towhee/tree/branch0.9.0/towhee/models/mcprop), 
[*RepLKNet*](https://github.com/towhee-io/towhee/tree/branch0.9.0/towhee/models/replknet), 
[*Shunted Transformer*](https://github.com/towhee-io/towhee/tree/branch0.9.0/towhee/models/shunted_transformer)
* Add two code search operators:
[*code_search.codebert*](https://towhee.io/code-search/codebert), 
[*code_search.unixcoder*](https://towhee.io/code-search/unixcoder)
* Add five image captioning operators: 
[*image_captioning.expansionnet-v2*](https://towhee.io/image-captioning/expansionnet-v2), 
[*image_captioning.magic*](https://towhee.io/image-captioning/magic),
[*image_captioning.clip_caption_reward*](https://towhee.io/image-captioning/clip-caption-reward), 
[*image_captioning.blip*](https://towhee.io/image-captioning/blip), 
[*image_captioning.clipcap*](https://towhee.io/image-captioning/clipcap)
* Add five image-text embedding operators: 
[*image_text_embedding.albef*](https://towhee.io/image-text-embedding/albef), 
[*image_text_embedding.ru_clip*](https://towhee.io/image-text-embedding/ru-clip), 
[*image_text_embedding.japanese_clip*](https://towhee.io/image-text-embedding/japanese-clip),
[*image_text_embedding.taiyi*](https://towhee.io/image-text-embedding/taiyi),
[*image_text_embedding.slip*](https://towhee.io/image-text-embedding/slip)
* Add one machine-translation operator: 
[*machine_translation.opus_mt*](https://towhee.io/machine-translation/opus-mt)
* Add one filter-tiny-segments operator:
[*video-copy-detection.filter-tiny-segments*](https://towhee.io/video-copy-detection/filter-tiny-segments)
* Add an advanced tutorial for audio fingerprinting: 
[*Audio Fingerprint II: Music Detection with Temporal Localization*](https://github.com/towhee-io/examples/blob/main/audio/audio_fingerprint/audio_fingerprint_advanced.ipynb) (increased accuracy from 84% to 90%)

**v0.8.1 Sep. 30, 2022**

* Added four visual backbones:
[*ISC*](https://github.com/towhee-io/towhee/tree/branch0.8.1/towhee/models/isc),
[*MetaFormer*](https://github.com/towhee-io/towhee/tree/branch0.8.1/towhee/models/metaformer),
[*ConvNext*](https://github.com/towhee-io/towhee/tree/branch0.8.1/towhee/models/convnext),
[*HorNet*](https://github.com/towhee-io/towhee/tree/branch0.8.1/towhee/models/hornet)
* Add two video de-copy operators:
[*select-video*](https://towhee.io/video-copy-detection/select-video), 
[*temporal-network*](https://towhee.io/video-copy-detection/temporal-network)
* Add one image embedding operator specifically designed for image retrieval and video de-copy with SOTA performance on VCSL dataset:
[*isc*](https://towhee.io/image-embedding/isc)
* Add one audio embedding operator specified for audio fingerprint:
[*audio_embedding.nnfp*](https://towhee.io/audio-embedding/nnfp) (with pretrained weights)
* Add one tutorial for video de-copy: 
[*How to Build a Video Segment Copy Detection System*](https://github.com/towhee-io/examples/blob/main/video/video_deduplication/segment_level/video_deduplication_at_segment_level.ipynb)
* Add one beginner tutorial for audio fingerprint:
[*Audio Fingerprint I: Build a Demo with Towhee & Milvus*](https://github.com/towhee-io/examples/blob/main/audio/audio_fingerprint/audio_fingerprint_beginner.ipynb)


**v0.8.0 Aug. 16, 2022**

* Towhee now supports generating an Nvidia Triton Server from a Towhee pipeline, with aditional support for GPU image decoding.
* Added one audio fingerprinting model: 
[*nnfp*](https://github.com/towhee-io/towhee/tree/branch0.8.0/towhee/models/nnfp)
* Added two image embedding models: 
[*RepMLP*](https://github.com/towhee-io/towhee/tree/branch0.8.0/towhee/models/repmlp), [**WaveViT**](https://github.com/towhee-io/towhee/tree/branch0.8.0/towhee/models/wave_vit)

**v0.7.3 Jul. 27, 2022**
* Added one multimodal (text/image) model:
[*CoCa*](https://github.com/towhee-io/towhee/tree/branch0.7.3/towhee/models/coca).
* Added two video models for grounded situation recognition & repetitive action counting:
[*CoFormer*](https://github.com/towhee-io/towhee/tree/branch0.7.3/towhee/models/coformer),
[*TransRAC*](https://github.com/towhee-io/towhee/tree/branch0.7.3/towhee/models/transrac).
* Added two SoTA models for image tasks (image retrieval, image classification, etc.):
[*CVNet*](https://github.com/towhee-io/towhee/tree/branch0.7.3/towhee/models/cvnet),
[*MaxViT*](https://github.com/towhee-io/towhee/tree/branch0.7.3/towhee/models/max_vit)

**v0.7.1 Jul. 1, 2022**
* Added one image embedding model:
[*MPViT*](https://towhee.io/image-embedding/mpvit).
* Added two video retrieval models:
[*BridgeFormer*](https://towhee.io/video-text-embedding/bridge-former),
[*collaborative-experts*](https://towhee.io/video-text-embedding/collaborative-experts).
* Added FAISS-based ANNSearch operators: *to_faiss*, *faiss_search*.

**v0.7.0 Jun. 24, 2022**

* Added six video understanding/classification models:
[*Video Swin Transformer*](https://towhee.io/action-classification/video-swin-transformer), 
[*TSM*](https://towhee.io/action-classification/tsm), 
[*Uniformer*](https://towhee.io/action-classification/uniformer), 
[*OMNIVORE*](https://towhee.io/action-classification/omnivore), 
[*TimeSformer*](https://towhee.io/action-classification/timesformer), 
[*MoViNets*](https://towhee.io/action-classification/movinet).
* Added four video retrieval models:
[*CLIP4Clip*](https://towhee.io/video-text-embedding/clip4clip), 
[*DRL*](https://towhee.io/video-text-embedding/drl), 
[*Frozen in Time*](https://towhee.io/video-text-embedding/frozen-in-time), 
[*MDMMT*](https://towhee.io/video-text-embedding/mdmmt).

**v0.6.1  May. 13, 2022**

* Added three text-image retrieval models:
[*CLIP*](https://towhee.io/image-text-embedding/clip),
[*BLIP*](https://towhee.io/image-text-embedding/blip),
[*LightningDOT*](https://towhee.io/image-text-embedding/lightningdot).
* Added six video understanding/classification models from PyTorchVideo:
[*I3D*](https://towhee.io/action-classification/pytorchvideo),
[*C2D*](https://towhee.io/action-classification/pytorchvideo),
[*Slow*](https://towhee.io/action-classification/pytorchvideo),
[*SlowFast*](https://towhee.io/action-classification/pytorchvideo),
[*X3D*](https://towhee.io/action-classification/pytorchvideo),
[*MViT*](https://towhee.io/action-classification/pytorchvideo).

## Getting started

Towhee requires Python 3.6+. You can install Towhee via `pip`:

```bash
pip install towhee towhee.models
```

If you run into any pip-related install problems, please try to upgrade pip with `pip install -U pip`.

Let's try your first Towhee pipeline. Below is an example for how to create a CLIP-based cross modal retrieval pipeline.

The example needs towhee 1.0.0, which can be installed with `pip install towhee==1.0.0`, The latest usage [documentation](https://towhee.readthedocs.io/en/main/index.html).

```python

from glob import glob
from towhee import ops, pipe, DataCollection


# create image embeddings and build index
p = (
    pipe.input('file_name')
    .map('file_name', 'img', ops.image_decode.cv2())
    .map('img', 'vec', ops.image_text_embedding.clip(model_name='clip_vit_base_patch32', modality='image'))
    .map('vec', 'vec', ops.towhee.np_normalize())
    .map(('vec', 'file_name'), (), ops.ann_insert.faiss_index('./faiss', 512))
    .output()
)

for f_name in ['https://raw.githubusercontent.com/towhee-io/towhee/main/assets/dog1.png',
               'https://raw.githubusercontent.com/towhee-io/towhee/main/assets/dog2.png',
               'https://raw.githubusercontent.com/towhee-io/towhee/main/assets/dog3.png']:
    p(f_name)

# Delete the pipeline object, make sure the faiss data is written to disk. 
del p


# search image by text
decode = ops.image_decode.cv2('rgb')
p = (
    pipe.input('text')
    .map('text', 'vec', ops.image_text_embedding.clip(model_name='clip_vit_base_patch32', modality='text'))
    .map('vec', 'vec', ops.towhee.np_normalize())
    # faiss op result format:  [[id, score, [file_name], ...]
    .map('vec', 'row', ops.ann_search.faiss_index('./faiss', 3))
    .map('row', 'images', lambda x: [decode(item[2][0]) for item in x])
    .output('text', 'images')
)

DataCollection(p('a cat')).show()

```
<img src="assets/towhee_example.png" style="width: 60%; height: 60%">

Learn more examples from the [Towhee Examples](https://github.com/towhee-io/examples).

## Core Concepts

Towhee is composed of four main building blocks - `Operators`, `Pipelines`, `DataCollection API` and `Engine`.

- __Operators__: An operator is a single building block of a neural data processing pipeline. Different implementations of operators are categorized by tasks, with each task having a standard interface. An operator can be a deep learning model, a data processing method, or a Python function.

- __Pipelines__: A pipeline is composed of several operators interconnected in the form of a DAG (directed acyclic graph). This DAG can direct complex functionalities, such as embedding feature extraction, data tagging, and cross modal data analysis.

- __DataCollection API__: A Pythonic and method-chaining style API for building custom pipelines. A pipeline defined by the DataColltion API can be run locally on a laptop for fast prototyping and then be converted to a docker image, with end-to-end optimizations, for production-ready environments.

- __Engine__: The engine sits at Towhee's core. Given a pipeline, the engine will drive dataflow among individual operators, schedule tasks, and monitor compute resource usage (CPU/GPU/etc). We provide a basic engine within Towhee to run pipelines on a single-instance machine and a Triton-based engine for docker containers.

## Contributing

Writing code is not the only way to contribute! Submitting issues, answering questions, and improving documentation are just some of the many ways you can help our growing community. Check out our [contributing page](https://github.com/towhee-io/towhee/blob/main/CONTRIBUTING.md) for more information.

Special thanks goes to these folks for contributing to Towhee, either on Github, our Towhee Hub, or elsewhere:
<br><!-- Do not remove start of hero-bot --><br>
<img src="https://img.shields.io/badge/all--contributors-33-orange"><br>
<a href="https://github.com/AniTho"><img src="https://avatars.githubusercontent.com/u/34787227?v=4" width="30px" /></a>
<a href="https://github.com/Chiiizzzy"><img src="https://avatars.githubusercontent.com/u/72550076?v=4" width="30px" /></a>
<a href="https://github.com/GuoRentong"><img src="https://avatars.githubusercontent.com/u/57477222?v=4" width="30px" /></a>
<a href="https://github.com/NicoYuan1986"><img src="https://avatars.githubusercontent.com/u/109071306?v=4" width="30px" /></a>
<a href="https://github.com/Opdoop"><img src="https://avatars.githubusercontent.com/u/21202514?v=4" width="30px" /></a>
<a href="https://github.com/Tumao727"><img src="https://avatars.githubusercontent.com/u/20420181?v=4" width="30px" /></a>
<a href="https://github.com/YuDongPan"><img src="https://avatars.githubusercontent.com/u/88148730?v=4" width="30px" /></a>
<a href="https://github.com/binbinlv"><img src="https://avatars.githubusercontent.com/u/83755740?v=4" width="30px" /></a>
<a href="https://github.com/derekdqc"><img src="https://avatars.githubusercontent.com/u/11754703?v=4" width="30px" /></a>
<a href="https://github.com/dreamfireyu"><img src="https://avatars.githubusercontent.com/u/47691077?v=4" width="30px" /></a>
<a href="https://github.com/filip-halt"><img src="https://avatars.githubusercontent.com/u/81822489?v=4" width="30px" /></a>
<a href="https://github.com/fzliu"><img src="https://avatars.githubusercontent.com/u/6334158?v=4" width="30px" /></a>
<a href="https://github.com/gexy185"><img src="https://avatars.githubusercontent.com/u/103474331?v=4" width="30px" /></a>
<a href="https://github.com/hyf3513OneGO"><img src="https://avatars.githubusercontent.com/u/67197231?v=4" width="30px" /></a>
<a href="https://github.com/jaelgu"><img src="https://avatars.githubusercontent.com/u/86251631?v=4" width="30px" /></a>
<a href="https://github.com/jeffoverflow"><img src="https://avatars.githubusercontent.com/u/24581746?v=4" width="30px" /></a>
<a href="https://github.com/jingkl"><img src="https://avatars.githubusercontent.com/u/34296482?v=4" width="30px" /></a>
<a href="https://github.com/jinlingxu06"><img src="https://avatars.githubusercontent.com/u/106302799?v=4" width="30px" /></a>
<a href="https://github.com/junjiejiangjjj"><img src="https://avatars.githubusercontent.com/u/14136703?v=4" width="30px" /></a>
<a href="https://github.com/krishnakatyal"><img src="https://avatars.githubusercontent.com/u/37455387?v=4" width="30px" /></a>
<a href="https://github.com/omartarek206"><img src="https://avatars.githubusercontent.com/u/40853054?v=4" width="30px" /></a>
<a href="https://github.com/oneseer"><img src="https://avatars.githubusercontent.com/u/28955741?v=4" width="30px" /></a>
<a href="https://github.com/pravee42"><img src="https://avatars.githubusercontent.com/u/65100038?v=4" width="30px" /></a>
<a href="https://github.com/reiase"><img src="https://avatars.githubusercontent.com/u/5417329?v=4" width="30px" /></a>
<a href="https://github.com/shiyu22"><img src="https://avatars.githubusercontent.com/u/53459423?v=4" width="30px" /></a>
<a href="https://github.com/songxianj"><img src="https://avatars.githubusercontent.com/u/107831450?v=4" width="30px" /></a>
<a href="https://github.com/soulteary"><img src="https://avatars.githubusercontent.com/u/1500781?v=4" width="30px" /></a>
<a href="https://github.com/sre-ci-robot"><img src="https://avatars.githubusercontent.com/u/56469371?v=4" width="30px" /></a>
<a href="https://github.com/sutcalag"><img src="https://avatars.githubusercontent.com/u/83750738?v=4" width="30px" /></a>
<a href="https://github.com/wxywb"><img src="https://avatars.githubusercontent.com/u/5432721?v=4" width="30px" /></a>
<a href="https://github.com/zc277584121"><img src="https://avatars.githubusercontent.com/u/17022025?v=4" width="30px" /></a>
<a href="https://github.com/zengxiang68"><img src="https://avatars.githubusercontent.com/u/68835157?v=4" width="30px" /></a>
<a href="https://github.com/zhousicong"><img src="https://avatars.githubusercontent.com/u/7541863?v=4" width="30px" /></a>
<br><!-- Do not remove end of hero-bot --><br>

Looking for a database to store and index your embedding vectors? Check out [Milvus](https://github.com/milvus-io/milvus).
