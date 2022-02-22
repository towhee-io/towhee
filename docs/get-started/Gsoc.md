---
id: GSOC
title: GSOC
---



# 2022 GSoC Towhee Proposal

### About Towhee

![Towhee](https://towhee.io/assets/img/logo-title.png)



Towhee is a flexible, application-oriented framework for running machine learning (ML) pipelines. It is a Python project that aims to make greatly simple anything2vec, allowing everyone - from beginner developers to large organizations - to deploy complex pipelines with just a few lines of code.
It has become a common practice among the industry to drive data applications with neural network models. Applications involving embedding can be seen in all walks of life, such as product recommender system, copyright protection, data management, software security, new drug discovery, finance, insurance, etc. Despite the fact that the concept of "embed everything" has prevailed in academia and industry in recent years, how to embed unstructured data reasonably and efficiently based on business needs is still an urgent problem that has not yet been resolved. The idea of the Towhee project is to sweep out those obstacles and make MLops significantly easier.
Some of Towhee's key features include:

•	Easy ML for everyone: Run a machine learning pipeline with less than five lines of code.

•	Rich operators and pipelines: Collaborate and share pipelines with the open source community.

•	Automatic versioning: Our versioning mechanism for pipelines and operators ensures that you never run into dependency hell.

•	Support for fine-tuning models: Feed your dataset into our Trainer and get a new model in just a few easy steps.

•	Deploy to cloud (to be implemented): Ready-made pipelines can be deployed to the cloud with minimal effort.

### Contacting Towhee

You can:
•	Find us on GitHub.

•	Join our Slack Channel.

•	Join our Office Hours every Thursday from 4-5pm PST.

•	Follow us on Twitter.

•	Visit our Towhee hub.

### Getting started
Towhee hosts its source code on GitHub. Before installing or contributing to Towhee, you should have Python (3.6 or later) installed on your device.

### Contributing
Contributions of all kinds to Towhee are welcomed. You can contribute your code by fixing a bug, developing a new feature, or adding a new pipeline.
The Towhee community maintains a list of Good First Issues( click [HERE](https://github.com/towhee-io/towhee/labels/good%20first%20issue)) which are friendly to entry-level contributors. You can also file a new issue following the pre-constructed templates when you find anything interesting to work on.
Make sure to check our Contributing Guide before making any contribution. All contributions should be made in the form of pull request on GitHub. Please follow the instruction on submitting a Pull Request.
Install Towhee and run your first pipeline
You can easily install Towhee with pip. For instructions on other installations, check our Installation Guide. 
> pip install -U pip  # if you run into installation issues, try updating pip
> 
> pip install towhee
> 
> Open up a python3 terminal and run a pre-build pipeline to generate an image embedding.

> from towhee import pipeline
> 
> embedding_pipeline = pipeline('image-embedding')
> 
> embedding = embedding_pipeline(‘Image_path’)

Now you have successfully run your first embedding pipeline with Towhee. Your image embedding is stored in embedding. 
## Project Ideas

### Towhee Hosted API

Description: Create a server that hosts Towhee and all of its pipelines behind a RESTful API. This API should implement all of the login/logout as well as key and token management. A user using the Towhee API can also upload a piece of data and have the server automatically select an appropriate embedding pipeline. The first phase of this project involves building a single-instance version of the API. This API should have all key features of a well-developed API, including asynchronicity via async/await, user authentication, and session management (via API keys or tokens). The second phase of this project involves horizontally scaling the Towhee API via Kubernetes or another container orchestration engine.

Skills needed: Proficiency in Python and a familiarity with API development using Python (FastAPI, Flask, or Django).

Difficulty level: Hard 

References: https://spapas.github.io/2021/08/25/django-token-rest-auth/,

https://www.cloudsavvyit.com/3987/how-do-you-build-an-api-server

https://www.django-rest-framework.org,

Mentors: Krishna Katyal(krishna.katyal@zilliz.com), Filip Haltmayer(filip.haltmayer@zilliz.com),Frank Liu(frank.liu@zilliz.com) 
### Extend the Towhee training subframework

Description: Deep neural network models can be accessed in the Towhee framework via an abstraction called an Operator. Models can fine-tuned by providing Towhee with an existing model-based operator as well as a labeled training dataset (such as ImageNet or CIFAR) and a set of training hyperparameters. This method of training is good for many standard datasets, but falls short in other cases such as semi-supervised training methods. This project involves extending the Towhee training framework to include GAN, autoencoder, and RNN training. Each training type should be given its own Trainer class within the Towhee framework and tested extensively on different computer vision and natural language processing datasets.

Skills needed: Proficiency in Python and a general understanding of GANs, autoencoders, and RNNs.

Difficulty level: Hard

References: Jia, Y. et al. Caffe: Convolutional Architecture for Feast Feature Embedding. https://arxiv.org/abs/1408.5093
Goodfellow, I. et al. Generative Adversarial Networks. https://arxiv.org/abs/1406.2661
Bank, D. et al. Autoencoders. Autoencoders
Sherstinsky, A. et al. Fundamentals of RNN and LSTM networks. https://arxiv.org/abs/1808.03314

Mentors: Kyle He(junchen.he@zilliz.com), Junjie Jiang(junjie.jiang@zilliz.com)

