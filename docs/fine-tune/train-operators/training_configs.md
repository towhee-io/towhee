---
id: training-configs
title: Training Configs
---

You can always set up training configs directly in python scripts or with a yaml file.
Refer to TrainingConfig for more API details.

## 1. Write Configs

You can dump default training configs or write customized training configs to a yaml file.
```python
from towhee.trainer.training_config import dump_default_yaml, TrainingConfig

# Dump default training configs to yaml
dump_default_yaml('default_training_configs.yaml')

# Write your training configs to yaml
training_configs = TrainingConfig()
training_configs.save_to_yaml('my_training_configs.yaml')
```

## 2. Modify Configs

Instead of setting training configs in script, you can also read training configs from a yaml file built on top of `default_training_configs.yaml`.
```python
from towhee.trainer.training_config import TrainingConfig

# Option 1: set parameters when creating TrainingConfig
training_configs = TrainingConfig(
    output_dir='my_output',
    overwrite_output_dir=True,
    eval_strategy='epoch',
    eval_steps=None,
    batch_size=5,
    val_batch_size=-1,
    seed=42,
    epoch_num=3,
    dataloader_pin_memory=True,
    dataloader_drop_last=True,
    dataloader_num_workers=-1,
    lr=5e-05,
    metric='Accuracy',
    print_steps=1,
    load_best_model_at_end=False,
    early_stopping={'mode': 'max', 'monitor': 'eval_epoch_metric', 'patience': 4},
    model_checkpoint={'every_n_epoch': 1},
    tensorboard=None,
    loss='CrossEntropyLoss',
    optimizer='Adam',
    lr_scheduler_type='linear',
    warmup_ratio=0.0,
    warmup_steps=0,
    device_str=None,
    n_gpu=-1,
    sync_bn=False,
    freeze_bn=False
    )

# Option 2: modify configs after creation
training_configs = TrainingConfig()
training_configs.output_dir = 'my_output'

# Option 3: read from yaml after creation
training_configs = TrainingConfig()
training_configs.load_from_yaml('my_training_configs.yaml')
```

## 3. Special Configs


#### print_steps
When `print_steps=1`, training results of each step (or batch) will be printed out in screen instead of an epoch progress bar.
    
    2022-03-03 16:59:41,635 - 4310336896 - trainer.py-trainer:390 - WARNING: TrainingConfig(output_dir='my_output', overwrite_output_dir=True, eval_strategy='epoch', eval_steps=None, batch_size=5, val_batch_size=-1, seed=42, epoch_num=3, dataloader_pin_memory=True, dataloader_drop_last=True, dataloader_num_workers=-1, lr=5e-05, metric='Accuracy', print_steps=1, load_best_model_at_end=False, early_stopping={'mode': 'max', 'monitor': 'eval_epoch_metric', 'patience': 4}, model_checkpoint={'every_n_epoch': 1}, tensorboard=None, loss='CrossEntropyLoss', optimizer='Adam', lr_scheduler_type='linear', warmup_ratio=0.0, warmup_steps=0, device_str=None, n_gpu=-1, sync_bn=False, freeze_bn=False)
    
    epoch=1/3, global_step=1, epoch_loss=2.469155788421631, epoch_metric=0.20000000298023224
    epoch=1/3, global_step=2, epoch_loss=2.486016273498535, epoch_metric=0.20000000298023224
    epoch=1/3, global_step=3, epoch_loss=2.519146203994751, epoch_metric=0.20000000298023224
    epoch=1/3, global_step=4, epoch_loss=2.451723098754883, epoch_metric=0.20000000298023224
    epoch=1/3, eval_global_step=0, eval_epoch_loss=2.263216495513916, eval_epoch_metric=0.20000000298023224
    epoch=1/3, eval_global_step=1, eval_epoch_loss=2.1709983348846436, eval_epoch_metric=0.20000000298023224
    epoch=2/3, global_step=5, epoch_loss=1.2240798473358154, epoch_metric=0.20000000298023224
    epoch=2/3, global_step=6, epoch_loss=1.1725499629974365, epoch_metric=0.20000000298023224
    epoch=2/3, global_step=7, epoch_loss=1.2648464441299438, epoch_metric=0.20000000298023224
    epoch=2/3, global_step=8, epoch_loss=1.30061936378479, epoch_metric=0.15000000596046448
    epoch=2/3, eval_global_step=2, eval_epoch_loss=1.2398303747177124, eval_epoch_metric=0.0
    epoch=2/3, eval_global_step=3, eval_epoch_loss=1.2246357202529907, eval_epoch_metric=0.10000000149011612
    epoch=3/3, global_step=9, epoch_loss=1.501572847366333, epoch_metric=0.20000000298023224
    epoch=3/3, global_step=10, epoch_loss=1.365707516670227, epoch_metric=0.20000000298023224
    epoch=3/3, global_step=11, epoch_loss=1.2403526306152344, epoch_metric=0.13333334028720856
    epoch=3/3, global_step=12, epoch_loss=1.0921388864517212, epoch_metric=0.10000000149011612
    epoch=3/3, eval_global_step=4, eval_epoch_loss=1.0393352508544922, eval_epoch_metric=0.0
    epoch=3/3, eval_global_step=5, eval_epoch_loss=1.0277410745620728, eval_epoch_metric=0.10000000149011612
