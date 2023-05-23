# flat_map()

Creates a flat-map node with optional configuration to the pipeline. This node calls the specified function to create a list matching the output schema from the one matching the input schema and flattens the output list.

```Python
flat_map(input_schema, output_schema, fn, config=None)
```

## Parameters

- **input_schema** - str or tuple[str]
  -  A column name or all column names of the input list

  -  Each column name in the schema should be a string, containing alphanumerical characters and underscores.
- **output_schema** - str or tuple[str]
  -  A column name or all column names of the output list

  -  Each column name in the schema should be a string, containing alphanumerical characters and underscores.
- **fn** - Operator, lambda, or callable
  -  A function that is used to map an item in the input list into an item in the output list.

  -  It can be an operator from [Towhee Hub](https://towhee.io/tasks/operator), a lambda, or a callable function.
- **config** - dict or None
  -  Optional configuration for the current node.

  -  It defaults to `None` and can be a dictionary containing the configuration items. See [AutoConfig API](https://zilliverse.feishu.cn/wiki/wikcnZvOj9KRWA3xSTBTQEb05De) for details.

## Returns

A not-callable `Pipeline` object with this flat-map node appended

## Example

```Python
from towhee import pipe, ops

p = (pipe.input('a')
         .flat_map('a', 'b', lambda x: [e+10 for e in x])
         .output('b'))
res = p([1,2,3])
res.to_list() # return [[11], [12], [13]]
```