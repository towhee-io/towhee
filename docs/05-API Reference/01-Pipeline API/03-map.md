# map()

Creates a map node with optional configuration to the pipeline. This node calls the specified function to create a list matching the output schema from the one matching the input schema.

```Python
map(input_schema, output_schema, fn, config=None)
```

## Parameters

- **input_schema** - str or tuple[str]
  -  A column name or all column names of the input list

  -  Each column name in the schema should be a string, containing alphanumerical characters and underscores.
- **output_schema** - str or tuple[str]
  -  A column name or all column names of the output list

  -  Each column name in the schema should be a string, containing alphanumerical characters and underscores.
- **fn** - Operator, lambda, or callable
  -  A function that is used to map an item in the input list to an item in the output list.

  -  It can be an operator from [Towhee Hub](https://towhee.io/tasks/operator), a lambda, or a callable function.
- **config** - dict or None
  -  Optional configuration for the current node.

  -  It defaults to `None` and can be a dictionary containing the configuration items. See [AutoConfig API](https://zilliverse.feishu.cn/wiki/wikcnZvOj9KRWA3xSTBTQEb05De) for details.

## Returns

A not-callable `Pipeline` object with this map node appended

## Example

```Python
from towhee import pipe, ops

# run with lambda
p = pipe.input('a').map('a', 'b', lambda x: x+1).output('b')
p(1).get() # return [2]

# run with callable
def func(x):
    return x+1
p = pipe.input('a').map('a', 'b', func).output('b')
p(1).get() # return [2]

# run with the operator in hub
p = (pipe.input('path')
         .map('path', 'img', ops.towhee.image_decode())
         .output('img'))
p('https://github.com/towhee-io/towhee/raw/main/towhee_logo.png')
```