# input()

Creates the input node of a pipeline with the input schema set to the schema of the actual input.

```Python
input(*input_schema)
```

## Parameters

- **input_schema** - tuple[str]
  -  One or multiple column names in a tuple. You should use some or all of these column names to define the input schema of the next pipeline node.

  -  A column name in the schema should be a string, containing alphanumerical characters and underscores.

## Returns

A not-callable `Pipeline` object.

## Example

```Python
from towhee import pipe, ops
p = pipe.input('a', 'b', 'c')
```