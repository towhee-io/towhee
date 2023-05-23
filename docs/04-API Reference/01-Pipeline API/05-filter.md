# filter()

Creates a filter node with optional configuration to the pipeline. This node calls the specified functions to create an output list by selecting members from the input list based on the function results. Therefore, the length of the output schema should be the same as that of the input schema.

```Python
filter(input_schema, output_schema, filter_columns, fn, config=None)
```

## Parameters

- **input_schema** - str or tuple[str]
  -  A column name or all column names of the input list

  -  Each column name in the schema should be a string, containing alphanumerical characters and underscores.
- **output_schema** - str or tuple[str]
  -  A column name or all column names of the output list

  -  Each column name in the schema should be a string, containing alphanumerical characters and underscores.
- **filter_columns** - str or tuple[str]
  -  Name of a column or names or several columns to which the filter function applies
- **fn** - Operator, lambda, or callable
  -  A function that is used to create an output list by selecting members from the input list.

  -  It can be an operator from [Towhee Hub](https://towhee.io/tasks/operator), a lambda, or a callable function.
- **config** - dict or None
  -  Optional configuration for the current node.

  -  It defaults to `None` and can be a dictionary containing the configuration items. See [AutoConfig API](https://zilliverse.feishu.cn/wiki/wikcnZvOj9KRWA3xSTBTQEb05De) for details.

## Returns

A not-callable `Pipeline` object with this filter node appended

## Example

```Python
from towhee import pipe

# run with lambda
p = pipe.input('a').filter('a', 'b', 'a', lambda x: x>10).output('b')
p(1).get() # return 
p(11).get() # return [11]
```