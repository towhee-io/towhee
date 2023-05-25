# window_all()

Creates a window-all node with optional configuration to the pipeline. This node creates a window to include all the items in the input list, applies the function to the items in the window, and appends the result of the function to the output list.

```Python
window_all(input_schema, output_schema, fn, config=None)
```

## Parameters

- **input_schema** - str or tuple[str]
  -  A column name or all column names of the input list

  -  Each column name in the schema should be a string, containing alphanumerical characters and underscores.
- **output_schema** - str or tuple[str]
  -  A column name or all column names of the output list

  -  Each column name in the schema should be a string, containing alphanumerical characters and underscores.
- **fn** - Operator, lambda, or callable
  -  A function with all items in the input list as the input.

  -  It can be an operator from [Towhee Hub](https://towhee.io/tasks/operator), a lambda, or a callable function.
- **config** - dict or None
  -  Optional configuration for the current node.

  -  It defaults to `None` and can be a dictionary containing the configuration items. See [AutoConfig API](https://zilliverse.feishu.cn/wiki/wikcnZvOj9KRWA3xSTBTQEb05De) for details.

##  Returns

A not-callable `Pipeline` object with this window-all node appended

##  Example

```Python
from towhee import pipe
    
p = (pipe.input('n1', 'n2')
         .flat_map(('n1', 'n2'), ('n1', 'n2'), lambda x, y: [(a, b) for a, b in zip(x, y)])
         .window_all(('n1', 'n2'), ('s1', 's2'), lambda x, y: (sum(x), sum(y)))
         .output('s1', 's2'))

p([1, 2, 3, 4], [2, 3, 4, 5]).get() # return [10, 14]
```