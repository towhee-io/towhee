# output()

Creates the output node of a pipeline with the output schema set to the schema of the actual output. No other nodes can be added to the pipeline after the output node.

```Python
output(*output_schema)
```

## Parameters

- **output_schema** - tuple[str]
  -  Zero or multiple column names in a tuple.

  -  A column name in the schema should be a string, containing alphanumerical characters and underscores.

## Returns

A not-callable `Pipeline` object. 

## Example

```Python
from towhee import pipe
p = pipe.input('a').map('a', 'b', lambda x: x+1).output('b')
p(1).get() # return [2]
```