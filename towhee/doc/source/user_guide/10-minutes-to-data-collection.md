---
id: 10-minutes-to-data-collection
title: 10 minutes to DataCollection
authors:
  - name: Jie HOU
    url: https://github.com/reiase
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# 10 minutes to DataCollection

This is a short introduction to `DataCollection`, an unstructured data processing framework provided by `towhee`. More complex examples can be found in the GitHub repo of [towhee](https://github.com/towhee-io/towhee).

## Preparation

The latest version of `towhee` can be installed with:

````{tab} pip
```shell
$ pip install towhee
```
````
````{tab} python -m pip
```shell
$ python -m pip install towhee
```
````

After that, we can import `towhee` as follows;


```python
>>> import towhee

```

## Create a DataCollection

`DataCollection` is an enhancement to the built-in `list` in python. Creating a `DataCollection` from a `list`:

````{tab} DataCollection
```python
>>> dc = towhee.dc([0, 1, 2, 3])
>>> dc
[0, 1, 2, 3]

```
````
````{tab} python
```python
>>> dc = [0, 1, 2, 3]
>>> dc
[0, 1, 2, 3]

```
````

The behavior of `DataCollection` is designed to be the same as `list`, making it easy to understand and compatible with most of the popular data science toolkits:

```python
>>> dc = towhee.dc([0, 1, 2, 3])
>>> dc
[0, 1, 2, 3]

>>> dc[1]
1

>>> dc[3]
3

>>> dc[:2]
[0, 1]

>>> dc.append(4).append(5)
[0, 1, 2, 3, 4, 5]

```

## View Data 

We can take a quick look at the data by `head()`:

```python
>>> dc = towhee.dc([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,])
>>> dc.head(5)
[0, 1, 2, 3, 4]

```

If you are running a jupyter notebook, `show()` is suggested for a better interface:

```{code-cell} ipython3
---
tags: [hide-cell]
---
import towhee
dc = towhee.dc([0, 1, 2, 3])
```

```{code-cell} ipython3
dc.show(limit=5)
```

## Process Data

### Apply a function by `map()`

Apply a function to the entities in a `DataCollection` :

```python
>>> towhee.dc([0, 1, 2, 3, 4]).map(lambda x: x*2)
[0, 2, 4, 6, 8]

```

### Filter the data with `filter()`

Select some entities in a `DataCollection` with a filter
```python
>>> towhee.dc([0, 1, 2, 3, 4]).filter(lambda x: int(x%2)==0)
[0, 2, 4]

```

### Chained Function Invocation

`DataCollection` supports *method-chaining style programming*, making the code clean and fluent. 

````{tab} DataCollection
```python
>>> (
...   	towhee.dc([0, 1, 2, 3, 4])
...           .filter(lambda x: x%2==1)
...           .map(lambda x: x+1)
...           .map(lambda x: x*2)
... )
[4, 8]

```
````
````{tab} python functional
```python
>>> list(
...     map(
...         lambda x: x*2,
...         map(lambda x: x+1,
...             filter(lambda x: x%2==1,
...                    [0, 1, 2, 3, 4])
...         )
...     )
... )
[4, 8]

```
````
````{tab} python for-loop
```python
>>> result = []
>>> for x in [0, 1, 2, 3, 4]:
...     if x%2 == 1:
...         x = x+1
...         x = x*2
...         result.append(x)
>>> result
[4, 8]

```
````

The code using `DataCollection` is more straightforward. Each action generates a new `DataCollection`, thus it can be followed by another action directly. 

## Use operators

`Operator` represents the basic unit of computation that is applied to the items in a `DataCollection`. 
There are many predefined Operators on [towhee hub](https://towhee.io), including the most popular deep learning model from computer vision and natural language processing.

### Towhee Operator

We can load an `Operator` from towhee hub as follows:

```python
>>> from towhee import ops
>>> op = ops.towhee.image_decode()
>>> img = op('./towhee_logo.png')

```

Where `towhee` is the namespace of the operator, and `image_decode` is the operator name. An operator is usually referred to with its full name: `namespace/name`. 

`towhee` is the namespace for official operators,
and also the default namespace if namespace is not specified:

```python
>>> from towhee import ops
>>> op = ops.image_decode()
>>> img = op('./towhee_logo.png')

```

### Custom Operator

It is also easy to define a custom operator with a function:

```python
>>> from towhee import register
>>> @register
... def add_1(x):
...     return x+1
>>> ops.add_1()(2)
3

```

If the operator needs additional initialize arguments, it has to be defined as a class:

```python
>>> @register
... class add_x:
...     def __init__(self, x):
...         self._x = x
...     def __call__(self, y):
...         return self._x + y

>>> ops.add_x(x=1)(2)
3

```

### Use named `Operator` with `DataCollection`

When an operator is uploaded to towhee hub or registered with `@register`,
we can call the operator with its name directly on a `DataCollection`:

```python
>>> @register
... def add_1(x):
...     return x+1

>>> (
...     towhee.dc([0, 1, 2, 3, 4])
...         .add_1()
... )
[1, 2, 3, 4, 5]

```

`add_1()` is an operator that registers to `towhee` with a decorator. We can invoke the operator as simple as calling a method of `DataCollection`. `DataCollection` will dispatch missing function calls to the operators.

Such call dispatch makes the code easy to read. Here is an example that compares the code using and not using call dispatch

````{tab} use call dispatch
```python
towhee.dc(some_image_list) \
    .image_decode() \
    .towhee.image_embedding(model_name='resnet50') \
    .tensor_normalize(axis=1)
```
````
````{tab} not use call dispatch
```python
towhee.dc(some_image_list) \
    .map(ops.image_decode()) \
    .map(ops.towhee.image_embedding(model_name='resnet50')) \
    .map(ops.tensor_normalize(axis=1))
```
````

## Stream Processing

For large-scale datasets, using a `list` is too memory-consuming for having to load the entire dataset into the memory. And the users may prefer stream processing with `generator`, which is offered by python and support to proceed only one item at once.

Towhee also provides streamed `DataCollection`.

### Create a streamed DataCollection

Streamed `DataCollection` is created from a generator.

```python
>>> dc = towhee.dc(iter([0, 1, 2, 3]))
>>> dc #doctest: +ELLIPSIS
<list_iterator object at ...>

```

### Use of streamed DataCollection

Streamed `DataCollection` is designed to behave the same as the unstreamed one. But we should notice that the computation will not run until we start to consume items from it.

````{tab} streamed
```python
>>> def debug_print(x):
...     print(f'debug print: {x}')
...     return x

>>> dc = ( #doctest: +ELLIPSIS
...   	towhee.dc(iter([0, 1, 2, 3, 4]))
...           .map(debug_print)
...           .filter(lambda x: x%2==1)
...           .map(lambda x: x+1)
...           .map(lambda x: x*2)
... )
>>> dc
<map object at 0x...>

>>> # consume the streamed dc and collection the result into a list
>>> [x for x in dc]
debug print: 0
debug print: 1
debug print: 2
debug print: 3
debug print: 4
[4, 8]

```
````
````{tab} unstreamed
```python
>>> def debug_print(x):
...     print(f'debug print: {x}')
...     return x

>>> dc = (
...   	towhee.dc([0, 1, 2, 3, 4])
...           .map(debug_print)
...           .filter(lambda x: x%2==1)
...           .map(lambda x: x+1)
...           .map(lambda x: x*2)
... )
debug print: 0
debug print: 1
debug print: 2
debug print: 3
debug print: 4
>>> dc
[4, 8]

```
````

In the example of streamed `DataCollection`, `debug_print()` is not executed until we start to access the items in the `DataCollection`. But for unstreamed `DataCollection`, it is executed immediately.

## Tabular Data

### Create a DataFrame with schema

### Apply Functions/Operators according to schema

## Advanced Features

