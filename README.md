![https://towhee.io](towhee_logo.png#gh-light-mode-only)
![https://towhee.io](towhee_logo_dark.png#gh-dark-mode-only)

<h3 align="center">
  <p style="text-align: center;"> <span style="font-weight: bold; font: Arial, sans-serif;">x</span>2vec, Towhee is all you need! </p>
</h3>

[![Slack](https://img.shields.io/badge/join-slack-orange?style=for-the-badge)](https://slack.towhee.io)
[![Twitter](https://img.shields.io/badge/follow-twitter-blue?style=for-the-badge)](https://twitter.com/towheeio)
[![License](https://img.shields.io/badge/license-apache2.0-green?style=for-the-badge)](https://www.apache.org/licenses/LICENSE-2.0)
[![Github Actions](https://img.shields.io/github/workflow/status/towhee-io/towhee/Workflow%20for%20pylint/main?label=pylint&style=for-the-badge)](https://github.com/towhee-io/towhee/actions/workflows/pylint.yml)
[![Coverage](https://img.shields.io/codecov/c/github/towhee-io/towhee?style=for-the-badge)](https://app.codecov.io/gh/towhee-io/towhee)

## What is Towhee?

Towhee is a flexible, application-oriented framework for generating embedding vectors via a pipeline of ML models and other operations. It aims to make democratize `x2vec`, allowing everyone - from beginner developers to large organizations - to generate dense embeddings with just a few lines of code.

To accomplish this, we provide pre-built pipelines for a variety of tasks, including audio/music embeddings, image embeddings, celebrity recognition, and more. For a full list of pipelines, feel free to visit our [Towhee hub](https://towhee.io). Although our initial focus is on generating embeddings, Towhee can also be used to create and share generic machine learning pipelines such as this [image animation pipeline](https://towhee.io/towhee/anime-transfer).

## Key features

- __Easy embedding for everyone__: Run a local embedding pipeline with as little as three lines of code.

- __Rich operators and pipelines__: No more reinventing the wheel! Collaborate and share pipelines with the open source community.

- __Automatic versioning__: Our versioning mechanism for pipelines and operators ensures that you never run into [dependency hell](https://en.wikipedia.org/wiki/Dependency_hell).

- __Support for fine-tuning models__: Feed your dataset into our `Trainer` and get a new model in just a few easy steps.

- __Deploy to cloud__*: Ready-made pipelines can be deployed to the cloud with minimal effort.

Features marked with a star (\*) are on our roadmap and have not yet been implemented. Help is always appreciated, so come join our [Slack](https://slack.towhee.io) or check out our [docs](https://docs.towhee.io) for more information.

## Getting started

[Run in Colab](https://colab.research.google.com/github/towhee-io/towhee/blob/main/docs/02-Getting%20Started/quick-start.ipynb)

Towhee requires Python 3.6+. Towhee can be installed via `pip`:

```bash
% pip install -U pip  # if you run into installation issues, try updating pip
% pip install towhee
```

Towhee provides a variety of pre-built pipelines. For example, generating an image embedding can be done in as little as five lines of code:

```python
>>> from towhee import pipeline

# Our built-in image embedding pipeline takes
>>> embedding_pipeline = pipeline('image-embedding')
>>> embedding = embedding_pipeline('https://docs.towhee.io/img/logo.png')
```

Your image embedding is now stored in `embedding`. It's that simple.

For image datasets, the users can also build their own pipeline with the [`DataCollection`](https://towhee.readthedocs.io/en/branch0.6/data_collection/get_started.html) API:

```python
import towhee

towhee.glob('./*.jpg') \
      .image_decode() \
      .image_embedding.timm(model_name='resnet50') \
      .to_list()
```

where [`image_decode`](https://towhee.io/towhee/image-decode) and [`image_embedding.timm`](https://towhee.io/image-embedding/timm) are operators from [towhee hub](https://towhee.io). The method-chaining style programming interface also support [parallel execution](https://towhee.readthedocs.io/en/branch0.6/data_collection/get_started.html#parallel-execution) and [exception handling](https://towhee.readthedocs.io/en/branch0.6/data_collection/get_started.html#exception-handling).

## Dive deeper

If you find that one of our default embedding pipelines does not suit you, you can also specify a custom pipeline from the hub as follows:

```python
>>> embedding_pipeline = pipeline('towhee/image-embedding-convnext-base')
```

For a full list of supported pipelines, visit our [docs page](https://docs.towhee.io).

Custom machine learning pipelines can be defined in a YAML file or via the [`DataCollection`](https://towhee.readthedocs.io/en/branch0.6/data_collection/get_started.html) API. The first time you instantiate and use a pipeline, all Python functions, configuration files, and model weights are automatically downloaded from the Towhee hub. To ease the development process, pipelines which already exist in the local Towhee cache (`/$HOME/.towhee/pipelines`) will be automatically loaded:

```python
# This will load the pipeline defined at $HOME/.towhee/pipelines/fzliu/my-embedding-pipeline.yaml
>>> embedding_pipeline = pipeline('fzliu/my-embedding-pipeline')
```

### Architecture overview

Towhee is composed of three main building blocks - `Pipelines`, `Operators`, and a singleton `Engine`.

- __Pipeline__: A `Pipeline` is a single embedding generation task that is composed of several operators. Operators are connected together within the pipeline via a directed acyclic graph.

- __Operator__: An `Operator` is a single node within a pipeline. An operator can be a machine learning model, a complex algorithm, or a Python function. All files needed to run the operator are contained within a directory (e.g. code, configs, models, etc...).

- __DataCollection__: A pythonic and method-chaining style API that for building custom unstructured data processing pipelines. `DataCollection` is designed to behave as a python list or iterator, DataCollection is easy to understand for python users and is compatible with most popular data science toolkits. Function/Operator invocations can be chained one after another, making your code clean and fluent.

- __Engine__: The `Engine` sits at Towhee's core. Given a `Pipeline`, the `Engine` will drive dataflow between individual operators, schedule tasks, and monitor compute resource (CPU/GPU/etc) usage. We provide a basic `Engine` within Towhee to run pipelines on a single-instance machine - K8s and other more complex `Engine` implementations are coming soon.

For a deeper dive into Towhee and its architecture, check out the [Towhee docs](https://docs.towhee.io).

## Contributing

Remember that writing code is not the only way to contribute! Submitting issues, answering questions, and improving documentation are some of the many ways you can join our growing community. Check out our [contributing page](https://github.com/towhee-io/towhee/blob/main/CONTRIBUTING.md) for more information.

Special thanks goes to these folks for contributing to Towhee, either on Github, our Towhee Hub, or elsewhere:
<br><!-- Do not remove start of hero-bot --><br>
<img src="https://img.shields.io/badge/all--contributors-24-orange"><br>
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
<a href="https://github.com/junjiejiangjjj"><img src="https://avatars.githubusercontent.com/u/14136703?v=4" width="30px" /></a>
<a href="https://github.com/krishnakatyal"><img src="https://avatars.githubusercontent.com/u/37455387?v=4" width="30px" /></a>
<a href="https://github.com/omartarek206"><img src="https://avatars.githubusercontent.com/u/40853054?v=4" width="30px" /></a>
<a href="https://github.com/oneseer"><img src="https://avatars.githubusercontent.com/u/28955741?v=4" width="30px" /></a>
<a href="https://github.com/pravee42"><img src="https://avatars.githubusercontent.com/u/65100038?v=4" width="30px" /></a>
<a href="https://github.com/reiase"><img src="https://avatars.githubusercontent.com/u/5417329?v=4" width="30px" /></a>
<a href="https://github.com/shiyu22"><img src="https://avatars.githubusercontent.com/u/53459423?v=4" width="30px" /></a>
<a href="https://github.com/sre-ci-robot"><img src="https://avatars.githubusercontent.com/u/56469371?v=4" width="30px" /></a>
<a href="https://github.com/sutcalag"><img src="https://avatars.githubusercontent.com/u/83750738?v=4" width="30px" /></a>
<a href="https://github.com/wxywb"><img src="https://avatars.githubusercontent.com/u/5432721?v=4" width="30px" /></a>
<a href="https://github.com/zc277584121"><img src="https://avatars.githubusercontent.com/u/17022025?v=4" width="30px" /></a>
<a href="https://github.com/zhousicong"><img src="https://avatars.githubusercontent.com/u/7541863?v=4" width="30px" /></a>
<br><!-- Do not remove end of hero-bot --><br>

Looking for a database to store and index your embedding vectors? Check out [Milvus](https://github.com/milvus-io/milvus).
