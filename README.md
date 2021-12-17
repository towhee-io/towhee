![https://towhee.io](docs/towhee_logo.png)

<h3 align="center">
  <p style="text-align: center;"> <span style="font-weight: bold; font: Arial, sans-serif;">x</span>2vec, Towhee is all you need! </p>
</h3>

[![Slack](https://img.shields.io/badge/join-slack-orange?style=for-the-badge)](https://slack.towhee.io)
[![License](https://img.shields.io/badge/license-apache2.0-green?style=for-the-badge)](https://www.apache.org/licenses/LICENSE-2.0)
[![Language](https://img.shields.io/badge/language-python-blue?style=for-the-badge&logoColor=ffdd54)](https://www.python.org/)
[![Github Actions](https://img.shields.io/github/workflow/status/towhee-io/towhee/Workflow%20for%20pylint/main?label=pylint&style=for-the-badge)]()
[![Coverage](https://img.shields.io/codecov/c/github/towhee-io/towhee?style=for-the-badge)]()

## What is Towhee?

Towhee is an flexible, application-oriented framework for computing embedding vectors over unstructured data. It aims to make democratize `anything2vec`, allowing everyone - from beginner developers to large organizations - to train and deploy complex machine learning pipelines with just a few lines of code.

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

Custom machine learning pipelines can be defined in a YAML file or via a Spark-like high-level programming interface (coming soon &trade;). The first time you instantiate and use a pipeline, all Python functions, configuration files, and model weights are automatically downloaded from the Towhee hub. To ease the development process, pipelines which already exist in the local Towhee cache (`/$HOME/.towhee/pipelines`) will be automatically loaded:

```python
# This will load the pipeline defined at $HOME/.towhee/pipelines/fzliu/my-embedding-pipeline.yaml
>>> embedding_pipeline = pipeline('fzliu/my-embedding-pipeline')
```

#### Architecture overview

Towhee is composed of three main building blocks - `Pipelines`, `Operators`, and a singleton `Engine`.

- __Pipeline__: A `Pipeline` is a single embedding generation task that is composed of several operators. Operators are connected together within the pipeline via a directed acyclic graph.

- __Operator__: An `Operator` is a single node within a pipeline. An operator can be a machine learning model, a complex algorithm, or a Python function. All files needed to run the operator are contained within a directory (e.g. code, configs, models, etc...).

- __Engine__: The `Engine` sits at Towhee's core. Given a `Pipeline`, the `Engine` will drive dataflow between individual operators, schedule tasks, and monitor compute resource (CPU/GPU/etc) usage. We provide a basic `Engine` within Towhee to run pipelines on a single-instance machine - K8s and other more complex `Engine` implementations are coming soon.

For a deeper dive into Towhee and its architecture, check out our [documentation](https://docs.towhee.io).

#### Design concepts

Towhee was created with a few key design concepts in mind. We differentiate ourselves from other machine learning frameworks by making machine learning accessible to a variety of users. To do this, we base Towhee's architecture and design around these key features.

- __Application-oriented__: Instead of being "just another model hub", we provide full end-to-end machine learning pipelines. Each pipeline can make use of any number of machine learning models or Python functions in a variety of configurations - ensembles, flows, or any combination thereof.

- __Convenient__:  Towhee pipelines can be created to implement a variety of practical embedding tasks. We provide a number of pre-built pipelines on our [hub](https://hub.towhee.io).

- __Extensible__: Individual operators within each pipeline can be reconfigured and reused in different pipelines. Pipelines can be deployed anywhere you want - on your local machine, on a server with 4 GPUs, or even in the cloud.
