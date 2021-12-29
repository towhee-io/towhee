![https://towhee.io](towhee_logo.png)

<h3 align="center">
  <p style="text-align: center;"> <span style="font-weight: bold; font: Arial, sans-serif;">x</span>2vec, Towhee is all you need! </p>
</h3>

[![Slack](https://img.shields.io/badge/join-slack-orange?style=for-the-badge)](https://slack.towhee.io)
[![Twitter](https://img.shields.io/badge/follow-twitter-blue?style=for-the-badge)](https://twitter.com/towheeio)
[![License](https://img.shields.io/badge/license-apache2.0-green?style=for-the-badge)](https://www.apache.org/licenses/LICENSE-2.0)
[![Github Actions](https://img.shields.io/github/workflow/status/towhee-io/towhee/Workflow%20for%20pylint/main?label=pylint&style=for-the-badge)]()
[![Coverage](https://img.shields.io/codecov/c/github/towhee-io/towhee?style=for-the-badge)]()

## What is Towhee?

Towhee is a flexible, application-oriented framework for computing embedding vectors over unstructured data. It aims to make democratize `anything2vec`, allowing everyone - from beginner developers to large organizations - to train and deploy complex machine learning pipelines with just a few lines of code.

Towhee has pre-built pipelines for a variety of tasks, including audio/music embeddings, image embeddings, celebrity recognition, and more. For a full list of pipelines, feel free to visit our [Towhee hub](https://hub.towhee.io).

## Key features

- __Easy embedding for everyone__: Transform your data into vectors with less than five lines of code.

- __Rich operators and pipelines__: No more reinventing the wheel! Collaborate and share pipelines with the open source community.

- __Automatic versioning__: Our versioning mechanism for pipelines and operators ensures that you never run into [dependency hell](https://en.wikipedia.org/wiki/Dependency_hell).

- __Support for fine-tuning models__*: Feed your dataset into our `Trainer` and get a new model in just a few easy steps.

- __Deploy to cloud__*: Ready-made pipelines can be deployed to the cloud with minimal effort.


Features marked with a star (\*) are on our roadmap and have not yet been implemented. Help is always appreciated, so come join our [Slack](https://slack.towhee.io) or check out our [docs](https://docs.towhee.io) for more information.

## Getting started

Towhee requires Python 3.6+ and Pytorch 1.4.0+. Support for Tensorflow and scikit-learn models is coming soon. Towhee can be installed via `pip`:

```bash
% pip install -U pip  # if you run into installation issues, try updating pip
% pip install towhee
```

Towhee provides a variety of pre-built embedding pipelines. For example, generating an embedding can be done in as little as five lines of code:

```python
>>> from towhee import pipeline

# Use our in-built embedding pipeline
>>> img_path = 'towhee_logo.png'
>>> embedding_pipeline = pipeline('image-embedding')
>>> embedding = embedding_pipeline(img_path)
```

Your image embedding is now stored in `embedding`. It's that simple.

## Dive deeper

If you find that one of our default embedding pipelines does not suit you, you can also specify a custom pipeline from the hub as follows:

```python
>>> embedding_pipeline = pipeline('towhee/image-embedding-resnet101')
```

For a full list of supported pipelines, visit our [docs page]().

Custom machine learning pipelines can be defined in a YAML file or via a Spark-like high-level programming interface (coming soon &trade;). The first time you instantiate and use a pipeline, all Python functions, configuration files, and model weights are automatically downloaded from the Towhee hub. To ease the development process, pipelines which already exist in the local Towhee cache (`/$HOME/.towhee/pipelines`) will be automatically loaded:

```python
# This will load the pipeline defined at $HOME/.towhee/pipelines/fzliu/my-embedding-pipeline.yaml
>>> embedding_pipeline = pipeline('fzliu/my-embedding-pipeline')
```

### Architecture overview

Towhee is composed of three main building blocks - `Pipelines`, `Operators`, and a singleton `Engine`.

- __Pipeline__: A `Pipeline` is a single embedding generation task that is composed of several operators. Operators are connected together within the pipeline via a directed acyclic graph.

- __Operator__: An `Operator` is a single node within a pipeline. An operator can be a machine learning model, a complex algorithm, or a Python function. All files needed to run the operator are contained within a directory (e.g. code, configs, models, etc...).

- __Engine__: The `Engine` sits at Towhee's core. Given a `Pipeline`, the `Engine` will drive dataflow between individual operators, schedule tasks, and monitor compute resource (CPU/GPU/etc) usage. We provide a basic `Engine` within Towhee to run pipelines on a single-instance machine - K8s and other more complex `Engine` implementations are coming soon.

For a deeper dive into Towhee and its architecture, check out the [Towhee docs](https://docs.towhee.io).

## Contributing

Remember that writing code is not the only way to contribute! Submitting issues, answering questions, and improving documentation are some of the many ways you can join our growing community. Check out our [contributing page](https://github.com/towhee-io/towhee/blob/main/CONTRIBUTING.md) for more information.

Special thanks goes to these folks for contributing to Towhee, either on Github, our Towhee Hub, or elsewhere:
<br><!-- Do not remove start of hero-bot --><br>
<img src="https://img.shields.io/badge/all--contributors-16-orange"><br>
<a href="https://github.com/Chiiizzzy"><img src="https://avatars.githubusercontent.com/u/72550076?v=4" width="30px" /></a>
<a href="https://github.com/GuoRentong"><img src="https://avatars.githubusercontent.com/u/57477222?v=4" width="30px" /></a>
<a href="https://github.com/binbinlv"><img src="https://avatars.githubusercontent.com/u/83755740?v=4" width="30px" /></a>
<a href="https://github.com/derekdqc"><img src="https://avatars.githubusercontent.com/u/11754703?v=4" width="30px" /></a>
<a href="https://github.com/filip-halt"><img src="https://avatars.githubusercontent.com/u/81822489?v=4" width="30px" /></a>
<a href="https://github.com/fzliu"><img src="https://avatars.githubusercontent.com/u/6334158?v=4" width="30px" /></a>
<a href="https://github.com/jaelgu"><img src="https://avatars.githubusercontent.com/u/86251631?v=4" width="30px" /></a>
<a href="https://github.com/jeffoverflow"><img src="https://avatars.githubusercontent.com/u/24581746?v=4" width="30px" /></a>
<a href="https://github.com/jennyli-z"><img src="https://avatars.githubusercontent.com/u/93511422?v=4" width="30px" /></a>
<a href="https://github.com/junjiejiangjjj"><img src="https://avatars.githubusercontent.com/u/14136703?v=4" width="30px" /></a>
<a href="https://github.com/oneseer"><img src="https://avatars.githubusercontent.com/u/28955741?v=4" width="30px" /></a>
<a href="https://github.com/shiyu22"><img src="https://avatars.githubusercontent.com/u/53459423?v=4" width="30px" /></a>
<a href="https://github.com/sre-ci-robot"><img src="https://avatars.githubusercontent.com/u/56469371?v=4" width="30px" /></a>
<a href="https://github.com/sutcalag"><img src="https://avatars.githubusercontent.com/u/83750738?v=4" width="30px" /></a>
<a href="https://github.com/wxywb"><img src="https://avatars.githubusercontent.com/u/5432721?v=4" width="30px" /></a>
<a href="https://github.com/zhousicong"><img src="https://avatars.githubusercontent.com/u/7541863?v=4" width="30px" /></a>
<br><!-- Do not remove end of hero-bot --><br>
