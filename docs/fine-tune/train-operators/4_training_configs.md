---
id: training-configs
title: Training Configs
---

# Training Configs
You can always set up training configs directly in python scripts or with a yaml file. Refer to TrainingConfig for more API details.

# 1. Default Configs
You can dump default training configs or write customized training configs to a yaml file.




```python
from towhee.trainer.training_config import dump_default_yaml, TrainingConfig
default_config_file = 'default_training_configs.yaml'
dump_default_yaml(default_config_file)
```

You can open default_training_configs.yaml, and you can get the default config yaml structure like this:
```yaml
train:
    output_dir: ./output_dir
    overwrite_output_dir: true
    eval_strategy: epoch
    eval_steps:
    batch_size: 8
    val_batch_size: -1
    seed: 42
    epoch_num: 2
    dataloader_pin_memory: true
    dataloader_drop_last: true
    dataloader_num_workers: 0
    load_best_model_at_end: false
    freeze_bn: false
device:
    device_str:
    sync_bn: false
logging:
    print_steps:
learning:
    lr: 5e-05
    loss: CrossEntropyLoss
    optimizer: Adam
    lr_scheduler_type: linear
    warmup_ratio: 0.0
    warmup_steps: 0
callback:
    early_stopping:
        monitor: eval_epoch_metric
        patience: 4
        mode: max
    model_checkpoint:
        every_n_epoch: 1
    tensorboard:
        log_dir:
        comment: ''
metrics:
    metric: Accuracy
```
So the yaml file is corresponding to the TrainingConfig instance.


```python
training_configs = TrainingConfig().load_from_yaml(default_config_file)
print(training_configs)
training_configs.output_dir = 'my_test_output'
training_configs.save_to_yaml('my_test_config.yaml')
```

    TrainingConfig(output_dir='./output_dir', overwrite_output_dir=True, eval_strategy='epoch', eval_steps=None, batch_size=8, val_batch_size=-1, seed=42, epoch_num=2, dataloader_pin_memory=True, dataloader_drop_last=True, dataloader_num_workers=0, lr=5e-05, metric='Accuracy', print_steps=None, load_best_model_at_end=False, early_stopping={'monitor': 'eval_epoch_metric', 'patience': 4, 'mode': 'max'}, model_checkpoint={'every_n_epoch': 1}, tensorboard={'log_dir': None, 'comment': ''}, loss='CrossEntropyLoss', optimizer='Adam', lr_scheduler_type='linear', warmup_ratio=0.0, warmup_steps=0, device_str=None, sync_bn=False, freeze_bn=False)


Open my_test_config.yaml, and you will find `output_dir` is modified:
```yaml
train:
    output_dir: my_test_output
```
So there are 2 ways to set up the configs. One is using by class `TrainingConfig`, another is to overwrite the yaml file.

# 2.Setting by TrainingConfig
It's easy to set config using the TrainingConfig class. Just set the fields in TrainingConfig instance.
You can get each config field introduction easily by `get_config_help()`.


```python
from towhee.trainer.training_config import get_config_help
help_dict = get_config_help() # get config field introductions.
```

    - output_dir
    {'help': 'The output directory where the model predictions and checkpoints will be written.', 'category': 'train'}
    
    - overwrite_output_dir
    {'help': 'Overwrite the content of the output directory.Use this to continue training if output_dir points to a checkpoint directory.', 'category': 'train'}
    
    - eval_strategy
    {'help': 'The evaluation strategy. It can be `steps`, `epoch`, `eval_epoch` or `no`,', 'category': 'train'}
    
    - eval_steps
    {'help': 'Run an evaluation every X steps.', 'category': 'train'}
    
    - batch_size
    {'help': 'Batch size for training.', 'category': 'train'}
    
    - val_batch_size
    {'help': 'Batch size for evaluation.', 'category': 'train'}
    
    - seed
    {'help': 'Random seed that will be set at the beginning of training.', 'category': 'train'}
    
    - epoch_num
    {'help': 'Total number of training epochs to perform.', 'category': 'train'}
    
    - dataloader_pin_memory
    {'help': 'Whether or not to pin memory for DataLoader.', 'category': 'train'}
    
    - dataloader_drop_last
    {'help': 'Drop the last incomplete batch if it is not divisible by the batch size.', 'category': 'train'}
    
    - dataloader_num_workers
    {'help': 'Number of subprocesses to use for data loading.default 0 means that the data will be loaded in the main process.-1 means using all the cpu kernels,it will greatly improve the speed when distributed training.', 'category': 'train'}
    
    - lr
    {'help': 'The initial learning rate for AdamW.', 'category': 'learning'}
    
    - metric
    {'help': 'The metric to use to compare two different models.', 'category': 'metrics'}
    
    - print_steps
    {'help': 'if None, use the tqdm progress bar, otherwise it will print the logs on the screen every `print_steps`', 'category': 'logging'}
    
    - load_best_model_at_end
    {'help': 'Whether or not to load the best model found during training at the end of training.', 'category': 'train'}
    
    - early_stopping
    {'help': '.', 'category': 'callback'}
    
    - model_checkpoint
    {'help': '.', 'category': 'callback'}
    
    - tensorboard
    {'help': '.', 'category': 'callback'}
    
    - loss
    {'help': 'Pytorch loss in torch.nn package', 'category': 'learning'}
    
    - optimizer
    {'help': 'Pytorch optimizer Class name in torch.optim package', 'category': 'learning'}
    
    - lr_scheduler_type
    {'help': 'The scheduler type to use.eg. `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`', 'category': 'learning'}
    
    - warmup_ratio
    {'help': 'Linear warmup over warmup_ratio fraction of total steps.', 'category': 'learning'}
    
    - warmup_steps
    {'help': 'Linear warmup over warmup_steps.', 'category': 'learning'}
    
    - device_str
    {'help': 'None -> if there is a cuda env in the machine, it will use cuda:0, else cpu;`cpu` -> use cpu only;`cuda:2` -> use the No.2 gpu.', 'category': 'device'}
    
    - sync_bn
    {'help': 'will be work if device_str is `cuda`, the True sync_bn would make training slower but acc better.', 'category': 'device'}
    
    - freeze_bn
    {'help': 'will completely freeze all BatchNorm layers during training.', 'category': 'train'}
    


You can construct config by the construct function, or then modify you custom value.
```python
training_configs = TrainingConfig(
    xxx='some_value_xxx',
    yyy='some_value_yyy'
)
# or
training_configs.aaa='some_value_aaa'
training_configs.bbb='some_value_bbb'
```

# 3.Setting by yaml file
Your yaml file can be briefly with just some lines. You need not write the whole setting.
```yaml
train:
    output_dir: my_another_output
```
A yaml like this also works. Default values will be overwritten if not written.
There are some point you should pay attention.
- If a value is None in python, no value is required after the colon.
- If the value is `True`/`False` in python, it's `true`/`false` in yaml.
- If the field is `str` instance in python, no quotation marks required.
- If the field value is `dict` instance in python, start another line after the colon, each line after that is each key-value pair info.
```yaml
    early_stopping:
        monitor: eval_epoch_metric
        patience: 4
        mode: max
```
equals
```python
early_stopping = {
    'monitor': 'eval_epoch_metric',
    'patience': 4,
    'mode': 'max'
    }
```
in python.

