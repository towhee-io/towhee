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
    <img src="https://github.com/towhee-io/towhee/actions/workflows/pylint.yml/badge.svg" alt="github actions"/>
  </a>
  <a href="https://pypi.org/project/towhee/">
    <img src="https://img.shields.io/pypi/v/towhee?label=Release&color&logo=Python" alt="github actions"/>
  </a>
  <a href="https://app.codecov.io/gh/towhee-io/towhee">
    <img src="https://img.shields.io/codecov/c/github/towhee-io/towhee?style=flat" alt="coverage"/>
  </a>
</div>

&nbsp;

[Towhee](https://towhee.io) is a cutting-edge framework designed to streamline the processing of unstructured data through the use of Large Language Model (LLM) based pipeline orchestration. It is uniquely positioned to extract invaluable insights from diverse unstructured data types, including lengthy text, images, audio and video files. Leveraging the capabilities of generative AI and the SOTA deep learning models, Towhee is capable of transforming this unprocessed data into specific formats such as text, image, or embeddings. These can then be efficiently loaded into an appropriate storage system like a vector database. Developers can initially build an intuitive data processing pipeline prototype with user friendly Pythonic API, then optimize it for production environments.

🎨 Multi Modalities: Towhee is capable of handling a wide range of data types. Whether it's image data, video clips, text, audio files, or even molecular structures, Towhee can process them all. 

📃    LLM Pipeline orchestration:  Towhee offers flexibility to adapt to different Large Language Models (LLMs). Additionally, it allows for hosting open-source large models locally. Moreover, Towhee provides features like prompt management and knowledge retrieval, making the interaction with these LLMs more efficient and effective.

🎓 Rich Operators: Towhee provides a wide range of ready-to-use state-of-the-art models across five domains: CV, NLP, multimodal, audio, and medical. With over 140 models like BERT and CLIP and rich functionalities like video decoding, audio slicing, frame sampling,  and dimensionality reduction, it assists in efficiently building data processing pipelines. 

🔌   Prebuilt ETL Pipelines: Towhee offers ready-to-use ETL (Extract, Transform, Load) pipelines for common tasks such as Retrieval-Augmented Generation, Text Image search, and Video copy detection. This means you don't need to be an AI expert to build applications using these features. 
⚡️  High performance backend: Leveraging the power of the Triton Inference Server, Towhee can speed up model serving on both CPU and GPU using platforms like TensorRT, Pytorch, and ONNX. Moreover, you can transform your Python pipeline into a high-performance docker container with just a few lines of code, enabling efficient deployment and scaling.

🐍 Pythonic API: Towhee includes a Pythonic method-chaining API for describing custom data processing pipelines. We also support schemas, which makes processing unstructured data as easy as handling tabular data.

## Getting started

Towhee requires Python 3.7+. You can install Towhee via `pip`:

```bash
pip install towhee towhee.models
```

### Pipeline

### Pre-defined Pipeline

Towhee provides some pre-defined pipelines to help users quickly implement some functions. 
Currently implemented are: 
- [Sentence Embedding](https://towhee.io/tasks/detail/pipeline/sentence-similarity)
- [Image Embedding](https://towhee.io/tasks/detail/pipeline/text-image-search)
- [Video deduplication](https://towhee.io/tasks/detail/pipeline/video-copy-detection)
- [Question Answer with Docs](https://towhee.io/tasks/detail/pipeline/retrieval-augmented-generation)

All pipelines can be found on Towhee Hub. Here is an example of using the sentence_embedding pipeline: 

```python
from towhee import AutoPipes, AutoConfig
# get the built-in sentence_similarity pipeline
config = AutoConfig.load_config('sentence_embedding')
config.model = 'paraphrase-albert-small-v2'
config.device = 0
sentence_embedding = AutoPipes.pipeline('sentence_embedding', config=config)

# generate embedding for one sentence
embedding = sentence_embedding('how are you?').get()
# batch generate embeddings for multi-sentences
embeddings = sentence_embedding.batch(['how are you?', 'how old are you?'])
embeddings = [e.get() for e in embeddings]
```
### Custom pipelines 

If you can't find the pipeline you want in towhee hub, you can also implement custom pipelines through the towhee Python API. In the following example, we will create a cross-modal retrieval pipeline based on CLIP.
```python

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

# Flush faiss data into disk. 
p.flush()
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

DataCollection(p('puppy Corgi')).show()

```
<img src="assets/towhee_example.png" style="width: 60%; height: 60%">


## Core Concepts

Towhee is composed of four main building blocks - `Operators`, `Pipelines`, `DataCollection API` and `Engine`.

- __Operators__: An operator is a single building block of a neural data processing pipeline. Different implementations of operators are categorized by tasks, with each task having a standard interface. An operator can be a deep learning model, a data processing method, or a Python function.

- __Pipelines__: A pipeline is composed of several operators interconnected in the form of a DAG (directed acyclic graph). This DAG can direct complex functionalities, such as embedding feature extraction, data tagging, and cross modal data analysis.

- __DataCollection API__: A Pythonic and method-chaining style API for building custom pipelines, providing multiple data conversion interfaces: map, filter, flat_map, concat, window, time_window, and window_all. Through these interfaces, complex data processing pipelines can be built quickly to process unstructured data such as video, audio, text, images, etc.

- __Engine__: The engine sits at Towhee's core. Given a pipeline, the engine will drive dataflow among individual operators, schedule tasks, and monitor compute resource usage (CPU/GPU/etc). We provide a basic engine within Towhee to run pipelines on a single-instance machine and a Triton-based engine for docker containers.

## Resource
- TowheeHub: https://towhee.io/
- docs: https://towhee.readthedocs.io/en/latest/
- examples: https://github.com/towhee-io/examples

## Contributing

Writing code is not the only way to contribute! Submitting issues, answering questions, and improving documentation are just some of the many ways you can help our growing community. Check out our [contributing page](https://github.com/towhee-io/towhee/blob/main/CONTRIBUTING.md) for more information.

Special thanks goes to these folks for contributing to Towhee, either on Github, our Towhee Hub, or elsewhere:
<br><!-- Do not remove start of hero-bot --><br>
<img src="https://img.shields.io/badge/all--contributors-45-orange"><br>
<a href="https://github.com/3270939387"><img src="https://avatars.githubusercontent.com/u/133976770?v=4" width="30px" /></a>
<a href="https://github.com/AniTho"><img src="https://avatars.githubusercontent.com/u/34787227?v=4" width="30px" /></a>
<a href="https://github.com/Armaggheddon"><img src="https://avatars.githubusercontent.com/u/47779194?v=4" width="30px" /></a>
<a href="https://github.com/Chiiizzzy"><img src="https://avatars.githubusercontent.com/u/72550076?v=4" width="30px" /></a>
<a href="https://github.com/GuoRentong"><img src="https://avatars.githubusercontent.com/u/57477222?v=4" width="30px" /></a>
<a href="https://github.com/KizAE86"><img src="https://avatars.githubusercontent.com/u/146533028?v=4" width="30px" /></a>
<a href="https://github.com/NicoYuan1986"><img src="https://avatars.githubusercontent.com/u/109071306?v=4" width="30px" /></a>
<a href="https://github.com/NothingEverHappens"><img src="https://avatars.githubusercontent.com/u/216412?v=4" width="30px" /></a>
<a href="https://github.com/Opdoop"><img src="https://avatars.githubusercontent.com/u/21202514?v=4" width="30px" /></a>
<a href="https://github.com/Sharp-rookie"><img src="https://avatars.githubusercontent.com/u/62098006?v=4" width="30px" /></a>
<a href="https://github.com/Tumao727"><img src="https://avatars.githubusercontent.com/u/20420181?v=4" width="30px" /></a>
<a href="https://github.com/UncleLLD"><img src="https://avatars.githubusercontent.com/u/16642335?v=4" width="30px" /></a>
<a href="https://github.com/YuDongPan"><img src="https://avatars.githubusercontent.com/u/88148730?v=4" width="30px" /></a>
<a href="https://github.com/binbinlv"><img src="https://avatars.githubusercontent.com/u/83755740?v=4" width="30px" /></a>
<a href="https://github.com/derekdqc"><img src="https://avatars.githubusercontent.com/u/11754703?v=4" width="30px" /></a>
<a href="https://github.com/dreamfireyu"><img src="https://avatars.githubusercontent.com/u/47691077?v=4" width="30px" /></a>
<a href="https://github.com/emmanuel-ferdman"><img src="https://avatars.githubusercontent.com/u/35470921?v=4" width="30px" /></a>
<a href="https://github.com/eric9204"><img src="https://avatars.githubusercontent.com/u/90449228?v=4" width="30px" /></a>
<a href="https://github.com/filip-halt"><img src="https://avatars.githubusercontent.com/u/81822489?v=4" width="30px" /></a>
<a href="https://github.com/fzliu"><img src="https://avatars.githubusercontent.com/u/6334158?v=4" width="30px" /></a>
<a href="https://github.com/gexy185"><img src="https://avatars.githubusercontent.com/u/103474331?v=4" width="30px" /></a>
<a href="https://github.com/huan415"><img src="https://avatars.githubusercontent.com/u/37132274?v=4" width="30px" /></a>
<a href="https://github.com/hyf3513OneGO"><img src="https://avatars.githubusercontent.com/u/67197231?v=4" width="30px" /></a>
<a href="https://github.com/jaelgu"><img src="https://avatars.githubusercontent.com/u/86251631?v=4" width="30px" /></a>
<a href="https://github.com/jeffoverflow"><img src="https://avatars.githubusercontent.com/u/24581746?v=4" width="30px" /></a>
<a href="https://github.com/jingkl"><img src="https://avatars.githubusercontent.com/u/34296482?v=4" width="30px" /></a>
<a href="https://github.com/jinlingxu06"><img src="https://avatars.githubusercontent.com/u/106302799?v=4" width="30px" /></a>
<a href="https://github.com/junjiejiangjjj"><img src="https://avatars.githubusercontent.com/u/14136703?v=4" width="30px" /></a>
<a href="https://github.com/krishnakatyal"><img src="https://avatars.githubusercontent.com/u/37455387?v=4" width="30px" /></a>
<a href="https://github.com/lrk612"><img src="https://avatars.githubusercontent.com/u/131778006?v=4" width="30px" /></a>
<a href="https://github.com/omartarek206"><img src="https://avatars.githubusercontent.com/u/40853054?v=4" width="30px" /></a>
<a href="https://github.com/oneseer"><img src="https://avatars.githubusercontent.com/u/28955741?v=4" width="30px" /></a>
<a href="https://github.com/pravee42"><img src="https://avatars.githubusercontent.com/u/65100038?v=4" width="30px" /></a>
<a href="https://github.com/reiase"><img src="https://avatars.githubusercontent.com/u/5417329?v=4" width="30px" /></a>
<a href="https://github.com/sanbuphy"><img src="https://avatars.githubusercontent.com/u/96160062?v=4" width="30px" /></a>
<a href="https://github.com/shiyu22"><img src="https://avatars.githubusercontent.com/u/53459423?v=4" width="30px" /></a>
<a href="https://github.com/songxianj"><img src="https://avatars.githubusercontent.com/u/107831450?v=4" width="30px" /></a>
<a href="https://github.com/soulteary"><img src="https://avatars.githubusercontent.com/u/1500781?v=4" width="30px" /></a>
<a href="https://github.com/sre-ci-robot"><img src="https://avatars.githubusercontent.com/u/56469371?v=4" width="30px" /></a>
<a href="https://github.com/sutcalag"><img src="https://avatars.githubusercontent.com/u/83750738?v=4" width="30px" /></a>
<a href="https://github.com/wxywb"><img src="https://avatars.githubusercontent.com/u/5432721?v=4" width="30px" /></a>
<a href="https://github.com/xychu"><img src="https://avatars.githubusercontent.com/u/936394?v=4" width="30px" /></a>
<a href="https://github.com/zc277584121"><img src="https://avatars.githubusercontent.com/u/17022025?v=4" width="30px" /></a>
<a href="https://github.com/zengxiang68"><img src="https://avatars.githubusercontent.com/u/68835157?v=4" width="30px" /></a>
<a href="https://github.com/zhousicong"><img src="https://avatars.githubusercontent.com/u/7541863?v=4" width="30px" /></a>
<br><!-- Do not remove end of hero-bot --><br>

Looking for a database to store and index your embedding vectors? Check out [Milvus](https://github.com/milvus-io/milvus).
