![https://towhee.io](docs/towhee_logo.png)

<h3 align="center">
  <p style="text-align: center;"> X2Vec, Towhee is all you need! </p>
</h3>

![Slack](https://img.shields.io/badge/join-slack-orange?style=for-the-badge)
![License](https://img.shields.io/badge/license-apache2.0-green?style=for-the-badge)
![Language](https://img.shields.io/badge/language-python-blue?style=for-the-badge&logoColor=ffdd54)
![Github Actions](https://img.shields.io/github/workflow/status/towhee-io/towhee/Workflow%20for%20pylint/main?label=pylint&style=for-the-badge)
![Coverage](https://img.shields.io/codecov/c/github/towhee-io/towhee?style=for-the-badge)

## What is Towhee?

Towhee is a flexible machine learning framework currently focused on computing deep learning embeddings over unstructured data. Built on top of PyTorch and Tensorflow (coming soon&trade;), Towhee provides a unified framework for running machine learning pipelines locally, on a multi-GPU/TPU/FPGA machine (coming soon&trade;), or in the cloud (coming soon&trade;). Towhee aims to make democratize machine learning, allowing everyone - from beginner developers to AI/ML research groups to large organizations - to train and deploy machine learning models.

## Key features

- __Easy embedding for everyone__: Transform your data into vectors with less than five lines of code.

- __Standardized pipeline__: Keep your pipeline interface consistent across projects and teams.

- __Rich operators and models__: No more reinventing the wheel! Collaborate and share models with the open source community.

- __Support for fine-tuning models__: Feed your dataset into our trainer and get a new model in just a few easy steps.

## Getting started

Towhee can be installed as follows:

```bash
% pip install -U pip
% pip cache purge
% pip install towhee
```

Towhee provides pre-built computer vision models which can be used to generate embeddings:

```python
>>> from towhee import pipeline
>>> from PIL import Image

# Use our in-built embedding pipeline
>>> img = Image.open('towhee_logo.png')
>>> embedding_pipeline = pipeline('image-embedding')
>>> embedding = embedding_pipeline(img)
```

Your image embedding is now stored in `embedding`. It's that simple.

Custom machine learning pipelines can be defined in a YAML file and uploaded to the Towhee hub (coming soon&trade;). Pipelines which already exist in the local Towhee cache (`/$HOME/.towhee/pipelines`) will be automatically loaded:

```python
# This will load the pipeline defined at $HOME/.towhee/pipelines/fzliu/resnet50_embedding.yaml
>>> embedding_pipeline = pipeline('fzliu/resnet50_embedding')
>>> embedding = embedding_pipeline(img)
```

## Dive deeper

#### Towhee architecture

- __Pipeline__: A `Pipeline` is a single machine learning task that is composed of several operators. Operators are connected together internally via a directed acyclic graph.

- __Operator__: An `Operator` is a single node within a pipeline. It contains files (e.g. code, configs, models, etc...) and works for reusable operations (e.g., preprocessing an image, inference with a pretrained model).

- __Engine__: The `Engine` sits at Towhee's core, and drives communication between individual operators, acquires and schedules tasks, and maintains CPU/GPU/FPGA/etc executors.

#### Design concepts

- __Flexible__: A Towhee pipeline can be created to implement any machine learning task you can think of.

- __Extensible__: Individual operators within each pipeline can be reconfigured and reused in different pipelines. A pipeline can be deployed anywhere you want - on your local machine, on a server with 4 GPUs, or in the cloud (coming soon&trade;)

- __Convenient__: Operators can be defined as a single function; new pipelines can be constructed by looking at input and output annotations for those functions. Towhee provides a high-level interface for creating new graphs by stringing together functions in Python code.
