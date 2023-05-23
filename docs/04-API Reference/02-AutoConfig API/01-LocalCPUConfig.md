# LocalCPUConfig()

Generates a configuration for running Towhee pipelines on the local CPU.

```Python
LocalCPUConfig()
```

## Returns

A `TowheeConfig` object with  `device` set to `-1`, indicating that the local CPU is used to run Towhee pipelines.

## Example

```Python
from towhee import pipe, AutoConfig

auto_config = AutoConfig.LocalCPUConfig()
auto_config.config # return {'device': -1}
```