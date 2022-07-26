&nbsp;

<p align="center">
    <img src="towhee_logo.png#gh-light-mode-only" width="60%"/>
    <img src="towhee_logo_dark.png#gh-dark-mode-only" width="60%"/>
</p>


<h3 align="center">
  <p style="text-align: center;"> 万物皆向量，一个 Towhee 即可！</p>
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

[Towhee](https://towhee.io) 让你轻松为 AI 应用搭建神经数据处理流水线。我们提供几百种模型、算法和转换任你选用，均可作为构建流水线的基本元素。Towhee 还提供了一套 Python 风格的 API，可以快速建立流水线 demo 并通过自动调优来适应生产环境。

:art:&emsp;**多模态数据**：Towhee 支持处理各种模态的数据，包括图像、视频、文本、音频乃至分子结构等等。

:mortar_board:&emsp;**SOTA 模型**：Towhee 提供涵盖 5 个领域（CV、NLP、多模态、音频、医疗）、15 种任务以及 140 多个模型架构的 SOTA 模型，包括BERT、CLIP、ViT、SwinTransformer、MAE 和 data2vec 等预训练模型，随时上手使用。

:package:&emsp;**数据处理**：Towhee 除了神经网络模型之外，还提供传统数据处理方法来帮助你建立实用的数据处理流水线。各类操作均可作为算子纳入流水线中，如视频解码、音频切片、帧采样、特征向量降维、集合和数据库操作等。

:snake:&emsp;**Python 风格 API**：Towhee 有一套 Python 风格的链式调用 API，用于自定义数据处理流水线。我们还支持数据库中常见的 schema，让非结构化数据处理起来像表格数据一样容易。

## 最近更新

**v0.7.1 2022/07/01**
* 新增支持 1 个图片 embedding 模型
[*MPViT*](https://towhee.io/image-embedding/mpvit)；
* 新增支持 2 个视频检索模型
[*BridgeFormer*](https://towhee.io/video-text-embedding/bridge-former),
[*collaborative-experts*](https://towhee.io/video-text-embedding/collaborative-experts)；
* 新增支持基于 FAISS 的 ANNSearch 算子：*to_faiss*, *faiss_search*。

**v0.7.0 2022/06/24**

* 新增支持 6 个视频理解/分类模型：
[*Video Swin Transformer*](https://towhee.io/action-classification/video-swin-transformer), 
[*TSM*](https://towhee.io/action-classification/tsm), 
[*Uniformer*](https://towhee.io/action-classification/uniformer), 
[*OMNIVORE*](https://towhee.io/action-classification/omnivore), 
[*TimeSformer*](https://towhee.io/action-classification/timesformer), 
[*MoViNets*](https://towhee.io/action-classification/movinet).
* 新增支持 4 个视频检索模型：
[*CLIP4Clip*](https://towhee.io/video-text-embedding/clip4clip), 
[*DRL*](https://towhee.io/video-text-embedding/drl), 
[*Frozen in Time*](https://towhee.io/video-text-embedding/frozen-in-time), 
[*MDMMT*](https://towhee.io/video-text-embedding/mdmmt).

**v0.6.1  2022/05/13**

* 新增支持 3 个图文检索模型：
[*CLIP*](https://towhee.io/image-text-embedding/clip),
[*BLIP*](https://towhee.io/image-text-embedding/blip),
[*LightningDOT*](https://towhee.io/image-text-embedding/lightningdot).
* 新增支持 6 个来自 PyTorchVideo 的视频理解/分类模型：
[*I3D*](https://towhee.io/action-classification/pytorchvideo),
[*C2D*](https://towhee.io/action-classification/pytorchvideo),
[*Slow*](https://towhee.io/action-classification/pytorchvideo),
[*SlowFast*](https://towhee.io/action-classification/pytorchvideo),
[*X3D*](https://towhee.io/action-classification/pytorchvideo),
[*MViT*](https://towhee.io/action-classification/pytorchvideo).

## 上手指南

Towhee 要求 Python 版本至少为 3.6，通过 `pip` 即可快速安装：

```bash
% pip install -U pip  # 如出现安装问题，建议尝试升级pip
% pip install towhee
```

安装完成之后即可开始使用 Towhee 搭建流水线。比如，你可以仅仅用15行代码就创建一个基于 CLIP 的跨模态数据检索流水线。

```python
import towhee

# 创建图片 embedding 并建立索引
(
    towhee.glob['file_name']('./*.png')
          .image_decode['file_name', 'img']()
          .image_text_embedding.clip['img', 'vec'](model_name='clip_vit_b32', modality='image')
          .tensor_normalize['vec','vec']()
          .to_faiss[('file_name', 'vec')](findex='./index.bin')
)

# 通过文字搜索图片
results = (
    towhee.dc['text'](['puppy Corgi'])
          .image_text_embedding.clip['text', 'vec'](model_name='clip_vit_b32', modality='text')
          .tensor_normalize['vec', 'vec']()
          .faiss_search['vec', 'results'](findex='./index.bin', k=3)
          .select['text', 'results']()
)
```
<img src="towhee_example.png" style="width: 60%; height: 60%">

更多用例可查阅 [Towhee Bootcamp](https://codelabs.towhee.io/)（英文内容）。

## 核心概念

Towhee 由四个核心成分组成：算子（operator）、流水线（pipeline）、DataCollection API 以及引擎。

- **算子**：算子是构成神经数据处理流水线的基本单元。算子的不同实现是按任务分类的，每个任务都有一个标准接口。算子可以是深度学习模型或是数据处理方法，甚至是 Python 函数。

- **流水线**：流水线由若干以 DAG （有向无环图）形式相互连接的算子组成。DAG 可以用来设计复杂功能，如嵌入特征提取、数据标记和跨模态数据分析等。

- **DataCollection API**: 这是一套 Python 风格的链式调用 API，用于构建自定义流水线。由 DataColltion API 定义的流水线可以在笔记本电脑上本地运行，用于快速制作原型，然后转换为 Docker 镜像，并通过端到端的优化用于生产环境。

- **引擎**：引擎是 Towhee 的核心，用于驱动流水线上各个算子之间的数据流、计划任务并监控计算资源的使用（CPU/GPU 等）。Towhee 提供了一个基本引擎（用于在单实例机器上运行流水线）以及一个基于 Triton 的引擎（用于 Docker 容器）。

## 贡献

写代码并不是唯一的贡献方式！我们欢迎开发者通过提交 Issue、回答问题、改进文档等各种方式向我们的社区做出贡献。具体请查看我们的 [贡献指南](https://github.com/towhee-io/towhee/blob/main/CONTRIBUTING.md)。

特别感谢以下开发者通过 GitHub、Towhee Hub 或其他方式为 Towhee 项目做出的贡献：
<br><!-- Do not remove start of hero-bot --><br>
<img src="https://img.shields.io/badge/all--contributors-27-orange"><br>
<a href="https://github.com/Chiiizzzy"><img src="https://avatars.githubusercontent.com/u/72550076?v=4" width="30px" /></a>
<a href="https://github.com/GuoRentong"><img src="https://avatars.githubusercontent.com/u/57477222?v=4" width="30px" /></a>
<a href="https://github.com/Tumao727"><img src="https://avatars.githubusercontent.com/u/20420181?v=4" width="30px" /></a>
<a href="https://github.com/binbinlv"><img src="https://avatars.githubusercontent.com/u/83755740?v=4" width="30px" /></a>
<a href="https://github.com/derekdqc"><img src="https://avatars.githubusercontent.com/u/11754703?v=4" width="30px" /></a>
<a href="https://github.com/filip-halt"><img src="https://avatars.githubusercontent.com/u/81822489?v=4" width="30px" /></a>
<a href="https://github.com/fzliu"><img src="https://avatars.githubusercontent.com/u/6334158?v=4" width="30px" /></a>
<a href="https://github.com/gexy185"><img src="https://avatars.githubusercontent.com/u/103474331?v=4" width="30px" /></a>
<a href="https://github.com/jaelgu"><img src="https://avatars.githubusercontent.com/u/86251631?v=4" width="30px" /></a>
<a href="https://github.com/jeffoverflow"><img src="https://avatars.githubusercontent.com/u/24581746?v=4" width="30px" /></a>
<a href="https://github.com/jennyli-z"><img src="https://avatars.githubusercontent.com/u/93511422?v=4" width="30px" /></a>
<a href="https://github.com/jingkl"><img src="https://avatars.githubusercontent.com/u/34296482?v=4" width="30px" /></a>
<a href="https://github.com/jinlingxu06"><img src="https://avatars.githubusercontent.com/u/106302799?v=4" width="30px" /></a>
<a href="https://github.com/junjiejiangjjj"><img src="https://avatars.githubusercontent.com/u/14136703?v=4" width="30px" /></a>
<a href="https://github.com/krishnakatyal"><img src="https://avatars.githubusercontent.com/u/37455387?v=4" width="30px" /></a>
<a href="https://github.com/omartarek206"><img src="https://avatars.githubusercontent.com/u/40853054?v=4" width="30px" /></a>
<a href="https://github.com/oneseer"><img src="https://avatars.githubusercontent.com/u/28955741?v=4" width="30px" /></a>
<a href="https://github.com/pravee42"><img src="https://avatars.githubusercontent.com/u/65100038?v=4" width="30px" /></a>
<a href="https://github.com/reiase"><img src="https://avatars.githubusercontent.com/u/5417329?v=4" width="30px" /></a>
<a href="https://github.com/shiyu22"><img src="https://avatars.githubusercontent.com/u/53459423?v=4" width="30px" /></a>
<a href="https://github.com/soulteary"><img src="https://avatars.githubusercontent.com/u/1500781?v=4" width="30px" /></a>
<a href="https://github.com/sre-ci-robot"><img src="https://avatars.githubusercontent.com/u/56469371?v=4" width="30px" /></a>
<a href="https://github.com/sutcalag"><img src="https://avatars.githubusercontent.com/u/83750738?v=4" width="30px" /></a>
<a href="https://github.com/wxywb"><img src="https://avatars.githubusercontent.com/u/5432721?v=4" width="30px" /></a>
<a href="https://github.com/zc277584121"><img src="https://avatars.githubusercontent.com/u/17022025?v=4" width="30px" /></a>
<a href="https://github.com/zhousicong"><img src="https://avatars.githubusercontent.com/u/7541863?v=4" width="30px" /></a>
<a href="https://github.com/zhujiming"><img src="https://avatars.githubusercontent.com/u/18031320?v=4" width="30px" /></a>
<br><!-- Do not remove end of hero-bot --><br>

需要一个能够存储并索引 embedding 向量的数据库吗？快来看看 [Milvus](https://github.com/milvus-io/milvus)吧。
