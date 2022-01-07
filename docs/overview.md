---
id: overview
title: Overview
sidebar_label: Overview
slug: /
---

### What is Towhee?

Towhee is a flexible, application-oriented framework for computing embedding vectors over unstructured data. It aims to make democratize `anything2vec`, allowing everyone - from beginner developers to large organizations - to train and deploy complex machine learning pipelines with just a few lines of code.

To accomplish this, we built Towhee atop popular machine learning and unstructured data processing libraries, i.e. PyTorch, scikit-learn, OpenCV. Models or functions from different libraries are wrapped as standard Towhee operators, and can be integrated freely into application-oriented pipelines. To ensure user-friendliness, pipelines can be called in just a single line of code, without the need to understand the underlying models or modules used to build it. For more information, take a look at our [quick start](get-started/quick-start) page.

### Problems Towhee solves

- **An embedding pipeline is far more than a single neural network.** Think about embedding items from a given video. Such embedding process will involve video decompression, key-frame extraction, frame deduplication, object detection, cropping, encoding, etc. For industrial practice, this necessitates a platform that offers end-to-end embedding pipeline solutions, as well as supporting data parallelism and resource management.

  Towhee solves this problem by reintroducing the concept of `Pipeline` as being _application-centric_ instead of _model-centric_. Where model-centric pipelines are composed of a single model followed by auxiliary code, application-centric pipelines treat every single data processing step as a first-class citizen.

- **Too many model implementations exist without any interface standard.** Machine learning models (NN-based and traditional) are ubiquitous. Different implementations of machine learning models require different auxiliary code to support testing and fine-tuning, making model evaluation and productionization a tedious task.

  Towhee solves this by providing a universal `Operator` wrapper for all models. Operators have a pre-defined API and glue logic to make Towhee work with a number of machine learning and data processing libraries.

- **MLOps is easier said than done.** Due to the continuous inflow of new training data, many DevOps teams now have a dedicated MLOps subteam to enable automated testing and productionization of machine learning models. The constant architectural updates to SOTA deep learning models also create significant overhead when deploying new said models in production environments.

  Towhee solves this by packaging model training, testing, and deployment in a single package. Any of our embedding generation pipelines can be deployed either on a laptop, across a multi-GPU server\*, or in a cluster of machines in just a couple lines of code\*.

\*_These features are coming in a future version of Towhee._

### Design philosophy

- **Convenient**: Towhee pipelines can be created to implement a variety of embedding tasks. Any pipeline creations or embedding tasks can be done in no more than 10 lines of code. We provide a number of pre-built pipelines on our [hub](https://towhee.io/pipelines?limit=30&page=1).

- **Extensible**: Individual operators have standard interfaces, and can be reconfigured/reused in different pipelines. Pipelines can be deployed anywhere you want - on your local machine, on a server with 4 GPUs, or even in the cloud.

- **Application-oriented**: Instead of being "just another model hub", we provide full end-to-end embedding pipelines. Each pipeline can make use of any number of machine learning models or Python functions in a variety of configurations - ensembles, flows, or any combination thereof.

### Where to go from here

#### Getting started:

- [Quick Start](get-started/quick-start): install Towhee and try your first pipeline.

#### Tutorials:

- [Reverse image search](tutorials/reverse-image-search): search for similar or related images.
- [Image deduplication](tutorials/image-deduplication): detect and remove identical or near-identical photos.
- [Music recognition](tutorials/music-recognition-system): music identification with full-length song or a snippet.

#### Supported pipelines:

- [Image embedding pipelines](pipelines/image-embedding)
- [Audio embedding pipelines](pipelines/audio-embedding)

#### Supported operators:

#### Community:

- Github: https://github.com/towhee-io/towhee
- Slack: https://slack.towhee.io
- Twitter: https://twitter.com/towheeio
