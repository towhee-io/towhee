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

[Towhee](https://towhee.io) 可以让用户像搭积木一样，轻松地完成 AI 应用程序的构建和落地。

通过使用神经网络数据处理流水线(neural data processing pipeline)的方式，我们可以将 Towhee 社区中数百个现成的模型、算法组合为标准的 AI 流水线。不仅如此，你还可以使用 Towhee 提供的 Pythonic API 来完成各种 AI 流水线和 AI 应用的原型设计，享受自动代码优化，低成本实现生产环境的应用性能优化。


## ✨ 项目特点

:art:&emsp;**多模态数据支持**：支持不同模态的数据处理，包括：图像、视频、文本、音频、分子结构等。

:mortar_board:&emsp;**SOTA 模型** 提供跨 5 个领域（CV、NLP、多模态、音频、医学）、15 种任务、140 个模型架构、700 个预训练的 SOTA 模型（例如：BERT、CLIP、ViT、SwinTransformer、MAE、data2vec 等）。

:package:&emsp;**数据处理** 除了神经网络模型，Towhee 同时提供了丰富的传统数据处理算子，包括：视频解码（Video decoding）、音频切片（audio slicing）、帧采样（frame sampling）、特征向量降维（feature vector dimension reduction）、模型融合（model ensemble）、数据库操作（database operations）等。配合各种模型、算法、数据处理方法，用户可以构建端到端的数据处理流水线。

:snake:&emsp;**Pythonic API:** 恪守 “Pythonic”，提供简洁、优雅、地道的 Python API。支持链式调用，能够快速定义富有表现力的数据处理流水线，让你处理非结构化数据和像处理表格数据一样简单。


## 📰 近期动态

**v1.0.0rc1 2023年5月4日**
* 新增一些模型支持微调。
[*timm*](https://towhee.io/image-embedding/timm), [*isc*](https://towhee.io/image-embedding/isc), [*transformers*](https://towhee.io/text-embedding/transformers), [*clip*](https://towhee.io/image-text-embedding/clip)
* 新增GPU视频解码算子: 
[*VPF*](https://towhee.io/video-decode/VPF)
* 所有的Pipeline均能够转换成Nvidia Triton 服务。

**v0.9.0 2022年12月2日**
* 新增一个视频分类模型:
[*Vis4mer*](https://github.com/towhee-io/towhee/tree/branch0.9.0/towhee/models/vis4mer)
* 新增三个视觉骨干模型:
[*MCProp*](https://github.com/towhee-io/towhee/tree/branch0.9.0/towhee/models/mcprop), 
[*RepLKNet*](https://github.com/towhee-io/towhee/tree/branch0.9.0/towhee/models/replknet), 
[*Shunted Transformer*](https://github.com/towhee-io/towhee/tree/branch0.9.0/towhee/models/shunted_transformer)
* 新增两个代码检索算子:
[*code_search.codebert*](https://towhee.io/code-search/codebert), 
[*code_search.unixcoder*](https://towhee.io/code-search/unixcoder)
* 新增五个图像描述生成算子: 
[*image_captioning.expansionnet-v2*](https://towhee.io/image-captioning/expansionnet-v2), 
[*image_captioning.magic*](https://towhee.io/image-captioning/magic),
[*image_captioning.clip_caption_reward*](https://towhee.io/image-captioning/clip-caption-reward), 
[*image_captioning.blip*](https://towhee.io/image-captioning/blip), 
[*image_captioning.clipcap*](https://towhee.io/image-captioning/clipcap)
* 新增五个图文跨模态特征提取算子: 
[*image_text_embedding.albef*](https://towhee.io/image-text-embedding/albef), 
[*image_text_embedding.ru_clip*](https://towhee.io/image-text-embedding/ru-clip), 
[*image_text_embedding.japanese_clip*](https://towhee.io/image-text-embedding/japanese-clip),
[*image_text_embedding.taiyi*](https://towhee.io/image-text-embedding/taiyi),
[*image_text_embedding.slip*](https://towhee.io/image-text-embedding/slip)
* 新增一个机器翻译算子: 
[*machine_translation.opus_mt*](https://towhee.io/machine-translation/opus-mt)
* 新增一个片段过滤算子:
[*video-copy-detection.filter-tiny-segments*](https://towhee.io/video-copy-detection/filter-tiny-segments)
* 新增一个音频指纹相关的进阶教程: 
[*Audio Fingerprint II: Music Detection with Temporal Localization*](https://github.com/towhee-io/examples/blob/main/audio/audio_fingerprint/audio_fingerprint_advanced.ipynb) (increased accuracy from 84% to 90%)

**v0.8.1 2022年9月30日**

* 新增四个视觉骨干模型:
[*ISC*](https://github.com/towhee-io/towhee/tree/branch0.8.1/towhee/models/isc),
[*MetaFormer*](https://github.com/towhee-io/towhee/tree/branch0.8.1/towhee/models/metaformer),
[*ConvNext*](https://github.com/towhee-io/towhee/tree/branch0.8.1/towhee/models/convnext),
[*HorNet*](https://github.com/towhee-io/towhee/tree/branch0.8.1/towhee/models/hornet)
* 新增两个视频去重算子:
[*select-video*](https://towhee.io/video-copy-detection/select-video), 
[*temporal-network*](https://towhee.io/video-copy-detection/temporal-network)
* 新增一个针对图片检索和视频去重的图像特征提取算子（在VCSL数据集上获得先进结果）:
[*isc*](https://towhee.io/image-embedding/isc)
* 新增一个针对音频指纹的音频特征提取算子:
[*audio_embedding.nnfp*](https://towhee.io/audio-embedding/nnfp) (with pretrained weights)
* 新增一个视频去重教程: 
[*How to Build a Video Segment Copy Detection System*](https://github.com/towhee-io/examples/blob/main/video/video_deduplication/segment_level/video_deduplication_at_segment_level.ipynb)
* 新增一个音频指纹相关的初级教程:
[*Audio Fingerprint I: Build a Demo with Towhee & Milvus*](https://github.com/towhee-io/examples/blob/main/audio/audio_fingerprint/audio_fingerprint_beginner.ipynb)


**v0.8.0 2022年8月16日**

* 支持“一键生成”基于 Nvidia Triton 的 Pipeline，快速构建适用于生产环节的 Docker Image，支持使用 GPU 对图片进行解码。
  * 教程：[“从定义流水线到生成 Docker 镜像，再到启动服务并调用执行，一共不到30行代码”](https://zhuanlan.zhihu.com/p/552917095)
* 新增一个音频指纹识别模型：[**nnfp**](https://github.com/towhee-io/towhee/tree/branch0.8.0/towhee/models/nnfp)
* 新增两个图片嵌入模型：[**RepMLP**](https://github.com/towhee-io/towhee/tree/branch0.8.0/towhee/models/repmlp), [**WaveViT**](https://github.com/towhee-io/towhee/tree/branch0.8.0/towhee/models/wave_vit)

**v0.7.3 2022年7月27日**

* 新增一个多模态（文本/图片）模型 :
[*CoCa*](https://github.com/towhee-io/towhee/tree/branch0.7.3/towhee/models/coca).
* 新增两个视频模型，用于“接地的态势感知（GSR）”和“智能计数重复动作”场景
Add two video models for grounded situation recognition & repetitive action counting:
[*CoFormer*](https://github.com/towhee-io/towhee/tree/branch0.7.3/towhee/models/coformer),
[*TransRAC*](https://github.com/towhee-io/towhee/tree/branch0.7.3/towhee/models/transrac).
* 新增两个图片任务（图像检索、图像分类等）的 SoTA 模型
[*CVNet*](https://github.com/towhee-io/towhee/tree/branch0.7.3/towhee/models/cvnet),
[*MaxViT*](https://github.com/towhee-io/towhee/tree/branch0.7.3/towhee/models/max_vit)

**v0.7.1 2022年7月1日**
* 新增一个图片嵌入模型（image embedding）:
[*MPViT*](https://towhee.io/image-embedding/mpvit).
* 新增两个视频检索模型（video retrieval）:
[*BridgeFormer*](https://towhee.io/video-text-embedding/bridge-former),
[*collaborative-experts*](https://towhee.io/video-text-embedding/collaborative-experts).
* 新增 FAISS-based ANNSearch 算子: *to_faiss*, *faiss_search*.

**v0.7.0 2022年6月24日**

* 新增六个视频理解/分类模型（video understanding/classification）
[*Video Swin Transformer*](https://towhee.io/action-classification/video-swin-transformer), 
[*TSM*](https://towhee.io/action-classification/tsm), 
[*Uniformer*](https://towhee.io/action-classification/uniformer), 
[*OMNIVORE*](https://towhee.io/action-classification/omnivore), 
[*TimeSformer*](https://towhee.io/action-classification/timesformer), 
[*MoViNets*](https://towhee.io/action-classification/movinet).
* 新增四个视频检索模型（video retrieval）
[*CLIP4Clip*](https://towhee.io/video-text-embedding/clip4clip), 
[*DRL*](https://towhee.io/video-text-embedding/drl), 
[*Frozen in Time*](https://towhee.io/video-text-embedding/frozen-in-time), 
[*MDMMT*](https://towhee.io/video-text-embedding/mdmmt).


**v0.6.1 2022年5月13日**

* 新增三个文本图像检索模型（text-image retrieval）：
[*CLIP*](https://towhee.io/image-text-embedding/clip),
[*BLIP*](https://towhee.io/image-text-embedding/blip),
[*LightningDOT*](https://towhee.io/image-text-embedding/lightningdot).
* 从 PyTorchVideo 添加六个视频理解/分类模型（video understanding/classification）：
[*I3D*](https://towhee.io/action-classification/pytorchvideo),
[*C2D*](https://towhee.io/action-classification/pytorchvideo),
[*Slow*](https://towhee.io/action-classification/pytorchvideo),
[*SlowFast*](https://towhee.io/action-classification/pytorchvideo),
[*X3D*](https://towhee.io/action-classification/pytorchvideo),
[*MViT*](https://towhee.io/action-classification/pytorchvideo).

## 🎓 快速入门

Towhee 需要 Python 3.6 及以上的运行环境，可以通过 `pip` 来完成快速安装：

```bash
pip install towhee towhee.models
```

安装就绪后，就能够创建你的第一个 AI 流水线啦。下面示例中，我们使用 15 行左右的代码，来创建一个基于 CLIP 的跨模态检索流水线。

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

for f_name in glob('./*.png'):
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

程序执行完毕，结果如下：

<img src="assets/towhee_example.png" style="width: 60%; height: 60%">

不够过瘾，想要了解更多例子吗？那么来👉 [Towhee 训练营](https://codelabs.towhee.io/) 👈 看看吧！

## 🚀 核心概念

Towhee 由四个主要模块组成：“算子（Operators）”、“流水线（Pipelines）”、“数据处理 API（DataCollection API）”和“执行引擎（Engine）”。

- __算子（Operator）__：算子是构成神经网络数据处理水流线(neural data processing pipeline)的“积木块”（基础组件）。这些基础组件按照任务类型进行组织，每种任务类型都具有标准的调用接口。一个算子可以是某种神经网络模型，某种数据处理方法，或是某个 Python 函数。

- __流水线（Pipeline）__：流水线是由若干个算子组成的 DAG（有向无环图）。流水线可以实现比单个算子更复杂的功能，诸如特征向量提取、数据标记、跨模态数据理解等。

- __数据处理 API（DataCollection）__: DataCollection API 是用于描述流水线的编程接口。基于 DataCollection 定义的流水线，既可以在 Jupyter Notebook 中本地运行，支持快速原型设计，也可以通过自动优化，一键构建出满足生产需要的高性能流水线服务，以及对应的 Docker 镜像。

- __执行引擎（Engine）__: 执行引擎负责实例化流水线、任务调度、资源管理，以及运行期性能优化。面向快速原型构建，Towhee 提供了轻量级的本地执行引擎；面向生产环境需求，Towhee 提供了基于 Nvidia Triton 的高性能执行引擎。

## 🏠 了解 & 加入社区

**编写代码并不是参与项目的唯一方式！**

你可以通过很多方式来参与 Towhee 社区：提交问题、回答问题、改进文档、加入社群讨论、参加线下 Meetup 活动等。

你的参与对于项目的持续健康发展至关重要。欢迎查阅 🎁[贡献页面](https://github.com/towhee-io/towhee/blob/main/CONTRIBUTING.md) 的文档内容，了解更多详细信息。

### 💥 致谢

特别感谢下面的同学为 Towhee 社区做出的贡献 🌹：

<br><!-- Do not remove start of hero-bot --><br>
<img src="https://img.shields.io/badge/all--contributors-32-orange"><br>
<a href="https://github.com/AniTho"><img src="https://avatars.githubusercontent.com/u/34787227?v=4" width="30px" /></a>
<a href="https://github.com/Chiiizzzy"><img src="https://avatars.githubusercontent.com/u/72550076?v=4" width="30px" /></a>
<a href="https://github.com/GuoRentong"><img src="https://avatars.githubusercontent.com/u/57477222?v=4" width="30px" /></a>
<a href="https://github.com/NicoYuan1986"><img src="https://avatars.githubusercontent.com/u/109071306?v=4" width="30px" /></a>
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

如果你正在寻找用于存储和检索向量的数据库，不妨看看[Milvus](https://github.com/milvus-io/milvus)。