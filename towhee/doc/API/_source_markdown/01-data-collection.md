# DataCollection Introduction

### What is `DataCollection`?

DataCollection is a pythonic computation framework for unstructured data in machine learning and data science.
It provides a method-chaining style API that makes it easier for a data scientist or researcher to assemble a data processing pipeline, finish his research on algorithms and models (embedding, transformer, or cnn) and apply the model to real-world business (search, recommendation, or shopping):

```python
>>> from towhee import DataCollection
>>> dc = DataCollection.range(5)
>>> (
...     dc.map(lambda x: x+1)
...       .map(lambda x: x*2)
...       .to_list()
... )
[2, 4, 6, 8, 10]

```

1. Pythonic Method-Chaining Style API: Designed to behave as a python list or iterator, DataCollection is easy to understand for python users and is compatible with most popular data science toolkits. Function/Operator invocations can be chained one after another, making your code clean and fluent.
2. Exception-Safe Execution: DataCollection provides exception-safe execution, which allows the function/operator invocation chain to continue executing on exception. Data scientists can put an exception receiver to the tail of the pipeline, processing and analyzing the exceptions as data, not errors.
2. Feature-Rich Operator Repository: There are various pre-defined operators On the [towhee hub](https://towhee.io), which cover the most popular deep learning models in computer vision, NLP, and voice processing. Using these operators in the data processing pipeline can significantly accelerate your work.

### Preparation

Please install `pandas`, `scikit-learn`, `opencv-python`, `ipython`, `matplotlib` to get all the functions of DataCollection.
```bash
$ pip install pandas scikit-learn scikit-learn opencv-python ipython matplotlib
```

### Quick Start

We use a `prime number` example to go through core conceptions in `DataCollection` and explain how the data and computation are organized. `prime numbers` are special numbers with exactly two factors, themselves and `1`. The following function detects if a number is prime:

```python
>>> def is_prime(x):
... 	if x <=1:
... 		return False
... 	for i in range(2, int(x/2)+1):
... 		if (x%i) == 0:
... 			return False
... 	return True

```

#### List or Iterator

`list` is the most widely used data structure in python, creating `DataCollection` from `list` is as simple as:

```python
>>> dc = DataCollection([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # use construction method

```

When the inputs are very large (for example, 1M), storing all the input in the memory is not a good idea. We can create `DataCollection` from an iterator for saving runtime memory:

```python
>>> dc = DataCollection(iter(range(10000))) # use construction method

```

### Functional Programming Interface

#### `map()`

After the `DataCollection` is created, we can apply `is_prime` to each element by `map(is_prime)`:

```python
>>> dc.map(is_prime) #doctest: +ELLIPSIS
<towhee.functional.data_collection.DataCollection object at 0x...>

```

`map()`  creates a new `DataCollection` that contains return values of `is_prime` on each element. To see the computation result, we need to convert the `DataCollection` back into a `list`:

```python
>>> dc = DataCollection.range(10)
>>> dc.map(is_prime).to_list()
[False, False, True, True, False, True, False, True, False, False]

```

#### `filter()`

Since the return value of `is_prime` is of the boolean type, we can filter the data collection with its return value and get a prime number list:

```python
>>> dc = DataCollection.range(10)
>>> dc.filter(is_prime).to_list()
[2, 3, 5, 7]

```

#### Method-Chaining coding style

A data processing pipeline can be created by chaining `map()` and `filter()`:

```python
>>> dc = (
... 	DataCollection.range(100)
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

1. `stage 2` will wait until all the calculations are done in `stage 1`;
2. A new `DataCollection` will hold all the outputs for each stage. You can perform list operations on the result `DataCollection`;

**NOTE**: `iterator-mode` and `stream-wise` computation

If the `DataCollection` is created from an iterator, it performs `stream-wise` computation and holds no data:

1. `DataCollection` takes one element from the input and applies `stage 1` and `stage 2` sequentially ;
2. Since `DataCollection` holds no data, indexing or shuffle is not supported;

We strongly suggest using `iterator-mode` instead of `list-mode` for memory efficiency and development convenience. When using `list-mode`, if the operator on the last stage is not appropriately implemented and runs into some error, you will waste all the computation in the previous stages.

---

### Operator dispatch

When chaining many stages, `DataCollection` provides a more straightforward syntax by *operator dispatching*:

```python
>>> from towhee import param_scope
>>> with param_scope() as hp:
...     hp().dispatcher.as_str = str
... 	dc = (
... 	   DataCollection.range(10)
... 		.filter(is_prime)
... 		.as_str() # map(str)
...     )

>>> dc.to_list()
['2', '3', '5', '7']

```

`DataCollection.as_str` is not defined in `DataCollection`. DataCollection will resolve such unknown functions by a lookup table managed with `param_scope`. By registering a new function via `param_scope().dispatcher`, we can extend `DataCollection` at run time without modifying the python code of `DataCollection`.

Operator dispatch is a fundamental mechanism in `DataCollection`. It allows `DataCollection` to be used as a DSL language for domain-specific tasks. For example, we can define an image processing pipeline as follows:

```python
dataset = DataCollection.from_glob('path_train/*.jpg') \
        .load_image() \
        .random_resize_crop(224) \
        .random_horizontal_flip() \
        .cvt_color('RGB2BGR') \
        .normalize()
```

This python code reads much more naturally for computer vision engineers and researchers.

Operator dispatch contains a hub-based resolve that loads operators from [towhee hub](https://towhee.io). This will accelerate your daily work with the help of the towhee community:

```python
dataset = DataCollection.from_glob('path_train/*.jpg') \
        .load_image() \
        .random_resize_crop(224) \
        .random_horizontal_flip() \
        .cvt_color('RGB2BGR') \
        .normalize() \
        .towhee.image_embedding(model_name='resnet101') # redirect function call to towhee operators
```

`.towhee.image_embedding(model_name='resnet101')` will be mapped to `towhee/image-embedding` on [towhee hub](https://towhee.io).

### Parallel execution

Dealing with large-scale datasets is usually time-consuming. `DataCollection` provides a simple API `set_parallel(num_worker)` to enable parallel execution, and speed up your computation :

```python
dc.set_parallel(num_worker=5) \
	.load_image() \
	.towhee.resnet_model(model_name='resnet50')
```

### Exception-handling

Most data scientists have experienced that exception crashes your code when dealing with large-scale datasets, damaged input data, unstable networks, or out of storage. It would be even worse if it crashes after having processed over half of the data, and you have to fix your code and re-run all the data.

`DataCollection` supports exception-safe execution, which can be enabled by a single line of code:

```python
dc.exception_safe() \ # or dc.safe()
	.load_image() \
	.towhee.resnet_model(model_name='resnet50')
```

When `exception_safe` is enabled, `DataCollection` generates empty values on exceptions. The lagging functions/operators will skip the empty values. And you can handle the empty values with an exception receiver at the tail of the chain:

```python
dc.safe() \
	.load_image() \
	.towhee.is_face_or_not() \
	.fill_empty(False) # or `.drop_empty()`
```

There are two builtin exception receivers:

1. `fill_empty(default_value)` will replace empty values with the default value;
2. `drop_empty(callback=None)` will drop the empty values and send the empty values to a callback function;

Here is an example callback function to receive exceptions from empty values:

```python
def callback(value):
  reason = value.get()
  error_input = reason.value
  exception = reason.exception
  ...
```
