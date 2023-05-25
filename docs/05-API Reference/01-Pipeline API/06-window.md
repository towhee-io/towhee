# window()

Creates a window node with optional configuration to the pipeline. This node iterates over the input list at the specified step size. Within each iteration, this node creates a window of the specified size in the input list, calls the function on the items in the window, and appends the result of the function to the output list.

```Python
window(input_schema, output_schema, size, step, fn, config=None)
```

## Parameters

- **input_schema** - str or tuple[str]
  -  A column name or all column names of the input list

  -  Each column name in the schema should be a string, containing alphanumerical characters and underscores.
- **output_schema** - str or tuple[str]
  -  A column name or all column names of the output list

  -  Each column name in the schema should be a string, containing alphanumerical characters and underscores.
- **size** - int
  -  Size of the window to be created within each iteration.
- **step** - int
  -  Step size of the iteration over the input list.
- **fn** - Operator, lambda, or callable
  -  A function with the items in the current window as the input.

  -  It can be an operator from [Towhee Hub](https://towhee.io/tasks/operator), a lambda, or a callable function.
- **config** - dict or None
  -  Optional configuration for the current node.

  -  It defaults to `None` and can be a dictionary containing the configuration items. See [AutoConfig API](https://zilliverse.feishu.cn/wiki/wikcnZvOj9KRWA3xSTBTQEb05De) for details.

## Returns

A not-callable pipeline object with this window node appended

## Example

```Python
from towhee import pipe

p = (pipe.input('n1', 'n2')
        .flat_map(('n1', 'n2'), ('n1', 'n2'), lambda x, y: [(a, b) for a, b in zip(x, y)])
        .window(('n1', 'n2'), ('s1', 's2'), 2, 1, lambda x, y: (sum(x), sum(y)))
        .output('s1', 's2'))
res = p([1, 2, 3, 4], [2, 3, 4, 5])
res.to_list() # return [[3, 5], [5, 7], [7, 9], [4, 5]]
```