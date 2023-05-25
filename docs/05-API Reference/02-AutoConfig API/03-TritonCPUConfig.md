# TritonCPUConfig()

Generates a configuration for running Towhee pipelines with a Triton inference server on the CPU. See [Towhee Pipeline in Triton](https://zilliverse.feishu.cn/wiki/wikcnvLGeh3znMQiuD4ENeBzvmh) for details.

```Python
TritonCPUConfig(num_instances_per_device=1, max_batch_size=None, batch_latency_micros=None, preferred_batch_size=None)
```

## Parameters

- **num_instances_per_device** - int
  -  Number of instances per CPU. 

  -  The value defaults to 1, indicating that there is one model instance running on the CPU.
- **max_batch_size** - int or None
  -  A maximum batch size that the model in the pipeline supports for the types of batching that can be exploited by Triton. See [Maximum Batch Size](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#maximum-batch-size) for details.

  -  The value defaults to `None`, leaving Triton to generate the value.
- **batch_latency_micros** - int or None
  -  Latency for Triton to process the delivered batch, in microseconds.

  -  The value defaults to `None`, leaving Triton to generate the value.
- **preferred_batch_size** - list[int] or None
  -  A list of batch sizes that the Triton should attempt to create.

  -  The value defaults to `None`, leaving Triton to generate the value.

## Returns

A `TowheeConfig` object with `server` set to a dictionary. The dictionary contains the specified parameters and their values with `device_ids` set to `None`.

## Examples

```Python
from towhee import pipe, ops, AutoConfig

auto_config1 = AutoConfig.TritonCPUConfig()
auto_config1.config # return {'server': {'device_ids': None, 'num_instances_per_device': 1, 'max_batch_size': None, 'batch_latency_micros': None, 'triton': {'preferred_batch_size': None}}}

# or you can also set the configuration
auto_config2 = AutoConfig.TritonCPUConfig(num_instances_per_device=3,
                                          max_batch_size=128,
                                          batch_latency_micros=100000,
                                          preferred_batch_size=[8, 16])
auto_config2.config # return {'server': {'device_ids': None, 'num_instances_per_device': 3, 'max_batch_size': 128, 'batch_latency_micros': 100000, 'triton': {'preferred_batch_size': [8, 16]}}}

# you can also add the configuration
auto_config3 = AutoConfig.LocalCPUConfig() + AutoConfig.TritonCPUConfig()
auto_config3.config # return {'device': -1, 'server': {'device_ids': None, 'num_instances_per_device': 1, 'max_batch_size': None, 'batch_latency_micros': None, 'triton': {'preferred_batch_size': None}}}
```