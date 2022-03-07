---
id: read-configs-from-yaml
title: Read the configs from a yaml file.
---

### Just start a jupyter-notebook and follow the steps below.

##### 1.clone the operator code.

```python
!git clone https://towhee.io/towhee/resnet-image-embedding.git
%cd resnet-image-embedding/
```

##### 2.build a resnet operator.


```python
import sys
# sys.path.append('..')
from resnet_image_embedding import ResnetImageEmbedding
from torchvision import transforms
from towhee import dataset

# build an resnet op:
op = ResnetImageEmbedding('resnet18', num_classes=10)
```

##### 3.dump a default config and adjust it.


```python
from towhee.trainer.training_config import dump_default_yaml

# If you want to see the default setting yaml, run dump_default_yaml()
dump_default_yaml('default_setting.yaml')
```

##### Then you can open `default_setting.yaml` to get the yaml structure.

##### Change `batch_size` to 5, `epoch_num` to 3, `tensorboard` to `null`, `output_dir` to `my_output`, `print_steps` to 1, and save it as `my_setting.yaml`

##### 4.read from your custom yaml.


```python
from towhee.trainer.training_config import TrainingConfig

training_config = TrainingConfig()
training_config.load_from_yaml('my_setting.yaml')
training_config

```


    TrainingConfig(output_dir='my_output', overwrite_output_dir=True, eval_strategy='epoch', eval_steps=None, batch_size=5, val_batch_size=-1, seed=42, epoch_num=3, dataloader_pin_memory=True, dataloader_drop_last=True, dataloader_num_workers=-1, lr=5e-05, metric='Accuracy', print_steps=1, load_best_model_at_end=False, early_stopping={'mode': 'max', 'monitor': 'eval_epoch_metric', 'patience': 4}, model_checkpoint={'every_n_epoch': 1}, tensorboard=None, loss='CrossEntropyLoss', optimizer='Adam', lr_scheduler_type='linear', warmup_ratio=0.0, warmup_steps=0, device_str=None, n_gpu=-1, sync_bn=False, freeze_bn=False)

##### 5.prepare the fake dataset


```python
fake_transform = transforms.Compose([transforms.ToTensor()])
train_data = dataset('fake', size=20, transform=fake_transform)
eval_data = dataset('fake', size=10, transform=fake_transform)
```

##### 6.start training,


```python
op.train(training_config, train_dataset=train_data, eval_dataset=eval_data)
```

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


##### Because you have set the `print_steps` to 1, you will not see the progress bar, instead, you will see the every batch steps result printed on the screen. You can check whether other configs ares work correctly.

##### By the way, you can change the config in your python code and save the config into a yaml file. So it's easy to convert between the python config instance and yaml file.


```python
training_config.batch_size = 2
training_config.save_to_yaml('another_setting.yaml')
```