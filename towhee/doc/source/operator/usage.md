# Operator
An operator is a single building block of a neural data processing pipeline. Different implementations of operators are categorized by tasks, with each task having a standard interface. An operator can be a deep learning model, a data processing method, or a Python function. All operators can be found on this page [https://towhee.io/operators](https://towhee.io/operators). This page classifies operators by category [https://towhee.io/tasks/operator](https://towhee.io/tasks/operator).

### Using Operators

We can load an `Operator` from the Towhee hub with the following:

```python
>>> from towhee import ops
>>> op = ops.towhee.image_decode()
>>> img = op('./towhee_logo.png')

```

Where `towhee` is the namespace of the operator, and `image_decode` is the operator name. An operator is usually referred to with its full name: `namespace/name`. 


### Custom Operators

It is also easy to define custom operators with standard Python functions:


```python
from towhee import register, ops
from towhee.operator import PyOperator

@register
class add_x(PyOperator):
    def __init__(self, x):
        self._x = x
    def __call__(self, y):
        return self._x + y

print(ops.add_x(1)(2))
# 3

```

### Using named `Operator`'s with `Pipeline`

When an operator is uploaded to the Towhee hub or registered with `@register`, We can use these operators in pipelines.:

```python
from towhee import register, ops, pipe
from towhee.operator import PyOperator

@register
class add_x(PyOperator):
    def __init__(self, x):
        self._x = x

    def __call__(self, y):
        return self._x + y

p = (
    pipe.input('num')
    .map('num', 'num', ops.add_x(10))
    .output('num')
    )
print(p(2).to_list())
# [[12]]

```
Use the operators in [towhee hub](https://towhee.io/tasks/operator).
```python
from towhee import register, ops, pipe
from towhee.operator import PyOperator

p = (
    pipe.input('file')
    .map('file', 'image', ops.image_decode.cv2('rgb'))
    .map('image', 'vec', ops.image_embedding.timm(model_name='resnet50'))
    .output('vec')
)

print(p('https://raw.githubusercontent.com/towhee-io/towhee/main/assets/dog1.png').to_list())
```

###  Using operators Directly.

```python
from towhee import ops
image_decode = ops.image_decode.cv2('rgb')
image_embedding = ops.image_embedding.timm(model_name='resnet50')

image_url = 'https://raw.githubusercontent.com/towhee-io/towhee/main/assets/dog1.png'
img = image_decode(image_url)
embedding = image_embedding(img)
print(embedding)
```



