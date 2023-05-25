# batch()

Runs a pipeline against multiple inputs in a batch.

```Python
batch(batch_inputs)
```

## Parameters

- **batch_inputs** - list
  -  A list of items, each of which matches the input schema of the pipeline.

## Returns

A list of items, each of which is the output of the pipeline against an input item.

## Example

```Python
from towhee import pipe
p = pipe.input('a').map('a', 'b', lambda x: x+1).output('b')
res = p.batch([1, 2, 3])
[r.get() for r in res] # return [[2], [3], [4]]
```