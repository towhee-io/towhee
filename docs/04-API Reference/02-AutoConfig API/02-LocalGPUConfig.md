# LocalGPUConfig()

Generates a configuration for running Towhee pipelines on the local GPU.

```Python
LocalGPUConfig(device)
```

## Parameters

- **device** - int
  -  ID of the GPU device to be used. The value defaults to 0, indicating that GPU 0 is used to run Towhee pipelines. 

  -  Setting this parameter to a non-existing GPU ID results in system errors. 

## Returns

A `TowheeConfig` object with `device` set to the specified value.

## Example

```Python
from towhee import pipe, ops, AutoConfig

auto_config = AutoConfig.LocalGPUConfig()
auto_config.config # return {'device': 0}
```