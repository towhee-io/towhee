# time_window()

Creates a time-window node with optional configuration to the pipeline. This node places each member element from the input list along a timeline according to the time point in the element's timestamp column. It then creates a window of the specified size every second, applies the function to the elements in the current window, and appends the result of the function to the output list.

```Python
time_window(input_schema, output_schema, timestamp_col, size, step, fn, config=None)
```

## Parameters

- **input_schema** - str or tuple[str]
  -  A column name or all column names of the input list

  -  Each column name in the schema should be a string, containing alphanumerical characters and underscores.
- **output_schema** - str or tuple[str]
  -  A column name or all column names of the output list

  -  Each column name in the schema should be a string, containing alphanumerical characters and underscores.
- **timestamp_col** - str
  -  Name of the timestamp column, indicating the position of the current element in the timeline.
- **size** - int
  -  Size of the window to be created within each iteration.
- **step** - int
  -  Step size of the iteration over the input list.
- **fn** - Operator, lambda, or callable
  -  A function with the items in the current timed window as the input.

  -  It can be an operator from [Towhee Hub](https://towhee.io/tasks/operator), a lambda, or a callable function.
- **config** - dict or None
  -  Optional configuration for the current node.

  -  It defaults to `None` and can be a dictionary containing the configuration items. See [AutoConfig API](https://zilliverse.feishu.cn/wiki/wikcnZvOj9KRWA3xSTBTQEb05De) for details.

## Returns

A not-callable `Pipeline` object with this time-window node appended

## Example

```Python
from towhee import pipe
p = (pipe.input('d')
        .flat_map('d', ('n1', 'n2', 't'), lambda x: ((a, b, c) for a, b, c in x))
        .time_window(('n1', 'n2'), ('s1', 's2'), 't', 3, 3, lambda x, y: (sum(x), sum(y)))
        .output('s1', 's2'))
data = [(i, i+1, i * 1000) for i in range(11) if i < 3 or i > 7]  # [(0, 1, 0), (1, 2, 1000), (2, 3, 2000), (8, 9, 8000), (9, 10, 9000), (10, 11, 10000)]
res = p(data)
res.to_list() # return [[3, 6], [8, 9], [19, 21]]
```