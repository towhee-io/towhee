---
id: hub-integration-and-caching
title: Hub integration and caching
---

A large part of what makes Towhee unique compared to other pipelining software is its emphasis on open source and integration with its own hub. Towhee hub allows for sharing and contributing to pipelines and operators that cover a wide range of use cases, ultimately allowing everyone to further their applications together. Lets go over how the hub works and how we ultimately integrate it into the framework. 


### Towhee Hub

**Implementation**

Towhee hub is deployed on a Gitea server. Gitea is an open-source, self-hosted git service. More information about Gitea can be found [here](https://gitea.io/en-us/).


**Integration**

All of Towhees communication with the hub is done through http requests. This was done in order to avoid requiring git installations on the system. 

Towhee operations that interact with the hub can be split into two categories, those that require authorization and those that do not. Lets begin with those that do not. 

**Non-Authorization**

The main operation that does not require authorization is downloading the repo. This request works by first obtaining the extensions that are stored in lfs. The next step is to download all the file names and group them by lfs and non-lfs downloads. This is required since LFS requires using a different http call to download the full data. Once we have these groups and file names, we can perform a multithreaded download.

**Authorized Requests**

Authorized requests require that you have login credentials setup with our hub. The main operations that require authorization and creating and deleting a repo. These requests use temporary tokens to speed up execution. 

## Caching

Caching was a complicated issue to solve with Towhee due to the fact that pipelines and operators might be stored in many different locations. Someone might want to deploy a locally saved pipeline that uses a hub based operator and two operators that are stored locally in separate locations. We solved this problem using a FileManager paired with a FileManagerConfig.

**FileManagerConfig**

Because Towhee is a multithreaded process, we needed a way to declare cache locations and pipelines/operators to be imported once and not have it change. We did this through using singletons. The FileManagerConfig is used to declare cache paths to look for files and also add locally saved pipelines/operators to the cache. Once a FileManager is created from the config, no changes or updates can be made.

**FileManager**

The file manager is in charge of loading all the required files. Once its created it first caches any local files that were declared in the Config. Cached files that are from the local system are assign the author 'local' so that they ignore all hub logic. After initialization, the file manager is used to get operators and pipeline files, with the option of redownloading them if the flag has been set. The FileManager is currently a singleton to avoid race conditions in downloading files.
