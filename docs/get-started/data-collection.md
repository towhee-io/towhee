---
id: datacollection
title: DataCollection Introduction
---

# What is DataCollection

DataCollection is a pythonic computation and processing framework for unstructured data in machine learning and data science. 
It allows a data scientist or researcher to assemble a data processing pipeline, do his model work (embedding, transforming, or classification) and apply it to the business (search, recommendation, or shopping) with a method-chaining style API.

Here is a short example for numerical computation with DataCollection:
```python
>>> dc = DataCollection.range(5)
>>> dc.map(lambda x: x+1) \
...   .map(lambda x: x*2*).to_list()
[2, 4, 6, 8. 10]
```

1. Pythonic Method-Chaining Style API: DataCollection is designed to be as easy as a python list or iterator. You can assemble a data processing pipeline by chaining functions/operations calls after a DataCollection.


2. Exception-Safe Execution: Handling exceptions in a multistage `pipeline` is usually painful for having to add a `try-catch` statement for each stage. DataCollection provides exception-safe execution, which allows the `pipeline` to continue executing on exception when processing large-scale datasets. The exceptions then proceed as data elements, not a workflow.


3. Feature-Rich Operator Repository: The [towhee hub](https://towhee.io) has various pre-defined operators, which can be chained into DataCollection's data processing pipeline. Operators on the towhee hub cover the most popular deep learning models in computer vision, NLP, and voice processing.

# Quick Start

We use a `prime number` example to go through core conceptions in `DataCollection`, and explain how the data and computation are organized. `prime numbers` are special numbers with exactly two factors, themselves and `1`. The following function detects whether a number is prime or not:

```python
>>> def is_prime(x):
... 	if x <=1:
... 		return False
... 	for i in range(2, int(x/2)+1):
... 		if (x%i) == 0:
... 			return False
... 	return True
```

## List or Iterator

`list` is the most widely used data structure in python, creating `DataCollection` from `list` is as simple as:

```python
>>> dc = DataCollection([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # use construction method
>>> dc = DataCollection.list(range(10)) # use factory method
```

When the inputs are very large (for example, 1M), storing all the input in the memory is not a good idea. We can create `DataCollection` from an iterator for saving runtime memory:

```python
>>> dc = DataCollection(iter(range(1000000))) # use construction method
>>> dc = DataCollection.iter(range(1000000)) # use factory method
```

## Functional Programming Interface

### `map()`

After the `DataCollection` is created, we can apply `is_prime` to each element by `map(is_prime)`:

```python
>>> dc.map(is_prime)
<towhee.functional.data_collection.DataCollection at 0x10732f4c0>
```

`map()`  creates a new `DataCollection` that contains return values of `is_prime` on each element. To see the computation result, we need to convert the `DataCollection` back into a `list`:

```python
>>> dc.map(is_prime).to_list()
[False, False, True, True, False, True, False, True, False, False]
```

### `filter()`

Since the return value of `is_prime` is of the boolean type, we can filter the data collection with its return value and get a prime number list:

```python
>>> dc.filter(is_prime).to_list()
[2, 3, 5, 7]
```

### `Method-Chaining Coding Style`

A data processing pipeline can be created by chaining `map()` and `filter()`:

```python
>>> dc = (
... 	DataCollection.iter(range(100))
... 		.filter(is_prime) # stage 1, find prime
... 		.filter(lambda x: x%10 == 3) # stage 2, find prime that ends with `3`
... 		.map(str) # stage 3, convert to string
... )
>>> dc.to_list()
['3', '13', '23', '43', '53', '73', '83']
```

---

**NOTE**: `list-mode` and `stage-wise` computation

When the `DataCollection` is created from a list, it will hold all the input values, and computations are performed in a `stage-wise` manner:

1. `stage 2` will wait until all the calculations are done in `stage 1;`
2. A new `DataCollection` will be created to hold all the outputs for each stage. You can perform list operations on result `DataCollection`;

**NOTE**: `iterator-mode` and `stream-wise` computation

If the `DataCollection` is created from an iterator, it performs `stream-wise` computation and holds no data:

1. `DataCollection` takes one element from the input and  applies `stage 1` and `stage 2` sequentially ;
2. Since `DataCollection` holds no data, indexing or shuffle is not supported;

We strongly suggest using `iterator-mode` instead of `list-mode`, for both memory efficiency and development convenience. When using `list-mode`, if the operator on the last stage is not appropriately implemented and runs into some error, you will waste all the computation in previous stages.

---

## Operator Dispatch

When chaining many stages, `DataCollection` provides an easier syntax:

```python
>>> with param_scope() as hp:
.. 		hp().dispatcher.as_str = str
... 	dc = (
... 	DataCollection.iter(range(10))
... 		.filter(is_prime)
... 		.as_str() # map(str)
... )
>>> dc.to_list()
['2', '3', '5', '7']
```

`DataCollection.as_str` is not defined in `DataCollection`. DataCollection will try to resolve this unknown function in a map when it is called. This map is managed with `param_scope` and can be updated at run time without modifying the python code of `DataCollection`. This is called `operator dispatch`, a fundamental mechanism that allows `DataCollection` to be used as a DSL language for domain-specific tasks. For example, an image processing pipeline can be defined as follows:

```python
>>> dataset = DataCollection.from_glob('path_train/*.jpg') \
        .load_image() \
        .random_resize_crop(224) \
        .random_horizontal_flip() \
        .cvt_color('RGB2BGR') \
        .normalize()
```

This python code reads much more naturally for computer vision engineers and researchers. 

Another essential feature of operator dispatch is that such function calls can be redirected to the towhee hub, by which you can greatly extend `DataCollection`'s abilities.

```python
>>> dataset = DataCollection.from_glob('path_train/*.jpg') \
        .load_image() \
        .random_resize_crop(224) \
        .random_horizontal_flip() \
        .cvt_color('RGB2BGR') \
        .normalize() \
        .towhee.image_embedding(model_name='resnet101') # redirect function call to towhee operators
```

`.towhee.image_embedding(model_name='resnet101')` will be mapped to `towhee/image-embedding`.

# Examples

This section will explain how to use `DataCollection` to complete your daily data science works.

## training imagenet
```python
>>> dataset = DataCollection.from_glob('path_train/*.jpg') \
        .load_image() \
        .random_resize_crop(224) \
        .random_horizontal_flip() \
        .to_tensor() \
        .normalize() \
        .to_pytorch()

>>> for i, data in enumerate(dataset):
...     inputs, labels = data
...     optimizer.zero_grad()
...     outputs = model(inputs)
...     loss = loss_fn(outputs, labels)
...     loss.backward()
...     optimizer.step()
```

## evaluating imagenet
```python
>>> summary = DataCollection.from_glob('path_eval/*.jpg')
...     .load_image() \
...     .img_resize(256) \
...     .img_center_crop(224) \
...     .to_tensor() \
...     .normalize() \
...     .apply_model(pytorch_model) \
...     .zip(labels) \
...     .summarize()
accuracy: 95%;
....
```

## Image Search
build image embeddings and load embeddings into milvus: 
```python
>>> DataCollection.from_glob('dataset_path/*.jpg') \
...     .load_image() \
...     .towhee.image_embedding(model_name='resnet50') \
...     .normalize() \
...     .to_milvus(uri='http+milvus://host:port/db/tbl') \
...     .run()
```

query image from milvus:
```python
>>> DataCollection(['query.jpg']) \
...     .load_image() \
...     .towhee.image_embedding(model_name='resnet50') \
...     .normalize() \
...     .search_from(uri='http+milvus://host:port/db/tbl') \
...     .to_list() 
```

## Face Detection and Recognition
```python
>>> DataCollection(['input_image.jph']) \
...     .load_image() \
...     .towhee.face_detection() \
...     .towhee.face_embedding() \
...     .search_from(uri='http+milvus://host:port/db/tbl', topk=1) \
...     .search_from(uri='redis://host:port/user_profiles') \
...     .to_list()
```

# Advanced Features

## Parallel Execution

## Exception-Safe Execution

