# concat()

Concatenates one or more pipelines to combine their output schemas.

```Python
concat(*pipeline)
```

## Parameters

- **pipeline** - tuple[Pipeline]
  -  A `Pipeline` object or multiple `Pipeline` objects in a tuple

## Returns

A not-callable pipeline object that combines the output schemas of all the pipelines involved

## Example

```Python
from towhee import pipe
pipe0 = pipe.input('a', 'b', 'c')
pipe1 = pipe0.map('a', 'd', lambda x: x+1)
pipe2 = pipe0.map(('b', 'c'), 'e', lambda x, y: x - y)
pipe3 = pipe2.concat(pipe1).output('d', 'e')
pipe3(1, 2, 3).get() # return [2, -1]
```