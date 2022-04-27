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

# DataCollection in 10 Minutes

This section is a short introduction to `DataCollection`, an unstructured data processing framework provided by `towhee`. More complex examples can be found in the Towhee [GitHub](https://github.com/towhee-io/towhee). 

## Preparation

The latest version of `towhee` can be installed with `pip`, or `python -m pip` if `pip` is not presented in your `PATH`:

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

With the package installed, we can import `towhee` with the following:


```python
>>> import towhee

```

## Creating a DataCollection

`DataCollection` is an enhancement to the built-in `list` in Python. Creating a `DataCollection` from a `list` is as simple as:

```python
>>> dc = towhee.dc([0, 1, 2, 3])
>>> dc
[0, 1, 2, 3]

```

The behavior of `DataCollection` is designed to be mimic `list`, making it easy to understand for most Python users and compatible with most of the popular data science toolkits;

```python
>>> dc = towhee.dc([0, 1, 2, 3])
>>> dc
[0, 1, 2, 3]

# indexing
>>> dc[1], dc[2]
(1, 2)

# slicing
>>> dc[:2]
[0, 1]

# appending
>>> dc.append(4).append(5)
[0, 1, 2, 3, 4, 5]

```

## Viewing Data 

We can take a quick look at the data by `head()`:

```python
>>> dc = towhee.dc([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,])
>>> dc.head(5)
[0, 1, 2, 3, 4]

```

If you are running within a jupyter notebook, `show()` is recommended as it provides a better interface:

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

## Processing Data

### Applying a Function

Applying a function to the elements in a `DataCollection` can be done with a simple `map()` call:

```python
>>> towhee.dc([0, 1, 2, 3, 4]).map(lambda x: x*2)
[0, 2, 4, 6, 8]

```

### Applying a Filter

Filtering the data in a `DataCollection`: 
```python
>>> towhee.dc([0, 1, 2, 3, 4]).filter(lambda x: int(x%2)==0)
[0, 2, 4]

```

### Chaining Data Processing Steps

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

The code using `DataCollection` is more straightforward, as each action generates a new `DataCollection`, thus allowing step by step instructions.

## Towhee Operators

Operators are the basic units of computation that can be applied to the elements within a `DataCollection`. 
There are many predefined Operators on the Towhee [hub](https://towhee.io), including popular deep learning models ranging from computer vision to natural language processing.

### Using Operators

We can load an `Operator` from the Towhee hub with the following:

```python
>>> from towhee import ops
>>> op = ops.towhee.image_decode()
>>> img = op('./towhee_logo.png')

```

Where `towhee` is the namespace of the operator, and `image_decode` is the operator name. An operator is usually referred to with its full name: `namespace/name`. 

`towhee` is the namespace for official operators, and also is the default namespace if not specified:

```python
>>> from towhee import ops
>>> op = ops.image_decode()
>>> img = op('./towhee_logo.png')

```

### Custom Operators

It is also easy to define custom operators with standard Python functions:

```python
>>> from towhee import register
>>> @register
... def add_1(x):
...     return x+1
>>> ops.add_1()(2)
3

```

If the operator needs additional initializations arguments, it can be defined as a class:

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

### Using named `Operator`'s with `DataCollection`

When an operator is uploaded to the Towhee hub or registered with `@register`, we can call the operato directly on a `DataCollection`:

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

`add_1()` is an operator that was registered to `towhee` using a decorator. We can invoke the operator by calling it as a method of `DataCollection`. `DataCollection` will dispatch missing function calls to the registered operators.

Such call dispatching makes the code easy to read. Here is code comparison of using call dispatch:

````{tab} using call dispatch
```python
towhee.dc(some_image_list) \
    .image_decode() \
    .towhee.image_embedding(model_name='resnet50') \
    .tensor_normalize(axis=1)
```
````
````{tab} not using call dispatch
```python
towhee.dc(some_image_list) \
    .map(ops.image_decode()) \
    .map(ops.towhee.image_embedding(model_name='resnet50')) \
    .map(ops.tensor_normalize(axis=1))
```
````
````{tab} without data collection
```python
image_decode = ops.image_decode()
image_embedding = ops.towhee.image_embedding(model_name='resnet50')
tensor_normalize = ops.tensor_normalize(axis=1)

result = []
for path in some_image_list:
  img = image_decode(path)
  embedding = image_embedding(img)
  vec = tensor_normalize(embedding)
  result.append(vec)
```
````

## Stream Processing

For large-scale datasets, using a `list` is too memory-intensive due to having to load the entire dataset into memory. Because of this, users often opt for stream processing with Python `generators`. These generators allow you to act on values as they come in, instead of having to wait for all the previous values to finish first before moving to the next step.

Towhee provides a similar streaming mechanism within `DataCollection`.

### Creating a Streamed DataCollection

A streamed `DataCollection` is created from a generator:

```python
>>> dc = towhee.dc(iter([0, 1, 2, 3]))
>>> dc #doctest: +ELLIPSIS
<list_iterator object at ...>

```

We can also convert an unstreamed `DataCollection` into a streamed one:

```python
>>> dc = towhee.dc([0, 1, 2, 3])
>>> dc.stream() #doctest: +ELLIPSIS
<list_iterator object at ...>
```

### Using Streamed DataCollections

Streamed `DataCollection`'s are designed to behave in the same way as the unstreamed ones. One important details is that the computation will not run until we begin consuming items from the `DataCollection`.

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

In the example of the streamed `DataCollection`, `debug_print()` is not executed until we start to access the items in the `DataCollection`. But for unstreamed `DataCollection`, it is executed immediately.

## Tabular Data

In this section we will introduce how to handle structured data with `DataCollection`. The term `tabular` refers to structured data that is organized into columns and rows, a widely used format by data scientists and supported by most machine learning toolkits.

### Creating a DataCollection with a Schema

- We can directly read data from files:

```python
dc = towhee.read_csv('some.csv')
dc = towhee.read_json('some.json')
```

- We can also load data from a pandas DataFrame:

```python
df = pandas.read_sql(...)
dc = towhee.from_df(df)
```

- We can also convert a list of `dict`s into a `DataCollection`:

```{code-cell} ipython3
>>> dc = towhee.dc([{'a': i, 'b': i*2} for i in range(5)]).as_entity()
>>> dc.show()
```

We call each row of the table an `Entity`. Both `a` and `b` are fields within the entity.

### Apply Functions/Operators according to schema

We can apply an operator according to the fields of the entities:

```{code-cell} ipython3
>>> @towhee.register
... def add_1(x):
...   return x+1

>>> dc.add_1['a', 'c']().show()
```

`['a', 'c']` is the syntax for specifying operator input and output,  field `a` is used as input, and field `c` is used as output. We can also apply a lambda function to tabular data with `runas_op`:

```{code-cell} ipython3
>>> dc.runas_op['b', 'd'](func=lambda x: x*2).show()
```

## Advanced Features

`DataCollection` also support advanced features such as parallel execution and distributed execution. To get more details about advanced feature, please refer to the API document of `DataCollection`. 
