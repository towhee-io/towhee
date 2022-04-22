---
id: data-collection-with-ray
title: Data Collection with Ray
authors:
  - name: Filip Haltmayer
    url: https://github.com/filip-halt
---

# What is Ray
[Ray](https://github.com/ray-project/ray) is python framework that simplifies the scaling and distributing of local workloads. On top of this, Ray provides many libraries for a wide range of ML workloads. In our case, we are using the Core API from Ray, as we require rays abilities in the lower levels of our code.


## How to Use
There are currently two ways to use the Ray backend while using the DataCollection API:

**1. Using the `dc.set_parallel(num_worker=n, backend='ray')` command.**

This command sets all subsequent parallizable commands to run parallized on Ray. `num_worker` generally
corresponds to how many workers are going to be assigned to the tasks. Due to different implementations
for different tasks, it isnt always a set in stone mapping. For example, in some instances, num workers
corresponds to how many active tasks should be running at once, while in another task it might mean how
many workers are assigned to the tasks.

An example of using set parallel is as follows:
```python
dc = DataCollection.range(1000).stream().set_parallel(num_worker=2, backend='ray')
a = dc.map(lambda x: x + 1)
b = a.map(lambda x: x * 2)
c = b.to_list()
```
In this example, both map() commands following the set_parallel() command will run on the ray backend.

> **_Note:_** Currently, the only supported functions that can run on the Ray backend are map() and operators from the [Towhee](https://towhee.io) hub.


**2. Using the `dc.pmap(... backend='ray')` and `dc.mmap(... backend='ray')`**

These commands are the parallel versions of map() and mmap(). With pmap(), you are using multiple machines to asynchronously calculate the map function results for the input. mmap() is the same, but instead of one function, you are running multiple functions, with each function outputting its own subsequent DataCollection.


## Custom Ray Connections
If using a custom ray connection, whether it be to a different machine or cluster, make sure to call ray.init() with the correct values before initializing the DataCollection chain. If it is not done, Ray will automatically run the commands locally.

> **_Note:_** When connecting to a ray cluster on a seperate machine, functions within the map() related calls that use third party libraries will not function. If using a custom function with custom dependencies, make sure to include those in the ray.init() call at the start of the program, more info can be found within [Rays documentation](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html).
