# load_config()

Loads the specified configuration for Towhee pipelines.

```Python
load_config(name, *args, **kwargs)
```

## Parameters

- **name** - str
  -  Name of the configuration. You can name it with some built-in pipelines, such as `sentence_embedding` or `text_image_embedding`.

  -  The value should be a string, containing alphanumerical characters and underscores.
- **args** - tuple
  -  Configuration items in the form of a tuple for some built-in pipelines.
- **kwargs** - dict   
  -  Configuration items in the form of a dictionary for some built-in pipelines.

## Returns

An `AutoConfig` object prefixed with the name of the configuration. For example, setting `name` to `sentence_embedding` results in the return of a `SentenceEmbeddingAutoConfig` object.

## Example

```Python
from towhee import AutoConfig

config = AutoConfig.load_config('sentence_embedding')
```