---
id: quick-start-training
title: Quick Start
---

Follow steps below to get started with a jupyter notebook for how to train a Towhee operator.
This example fine-tunes a pretrained model (eg. resnet-18) with a fake dataset.

## 1. Download Operator
Download operator files together with the jupyter notebook.
```bash
$ git clone https://towhee.io/towhee/resnet-image-embedding.git
$ cd resnet-image-embedding/examples
```
Then run Python scripts in following steps to train and test a Towhee operator.

## 2. Setup Operator
Create operator and load model by name.
```python
import sys
sys.path.append('..')

from resnet_image_embedding import ResnetImageEmbedding

op = ResnetImageEmbedding('resnet18', num_classes=10)
```

## 3. Configure Trainer:
Modify training configurations on top of default values.
```python
from towhee.trainer.training_config import TrainingConfig

training_config = TrainingConfig()
training_config.batch_size = 2
training_config.epoch_num = 2
training_config.tensorboard = None
training_config.output_dir = 'quick_start_output'
```

## 4. Prepare Dataset
The example here uses a fake dataset for both training and evaluation.
```python
from torchvision import transforms
from towhee import dataset

fake_transform = transforms.Compose([transforms.ToTensor()])
train_data = dataset('fake', size=20, transform=fake_transform)
eval_data = dataset('fake', size=10, transform=fake_transform)
```

## 5. Start Training
Now everything is ready, start training.

```python
op.train(
    training_config,
    train_dataset=train_data,
    eval_dataset=eval_data
    )
```

With a successful training, you will see progress bar below and a `quick_start_output` folder containing training results.

    2022-03-02 15:09:06,334 - 8665081344 - trainer.py-trainer:390 - WARNING: TrainingConfig(output_dir='quick_start_output', overwrite_output_dir=True, eval_strategy='epoch', eval_steps=None, batch_size=2, val_batch_size=-1, seed=42, epoch_num=2, dataloader_pin_memory=True, dataloader_drop_last=True, dataloader_num_workers=-1, lr=5e-05, metric='Accuracy', print_steps=None, load_best_model_at_end=False, early_stopping={'monitor': 'eval_epoch_metric', 'patience': 4, 'mode': 'max'}, model_checkpoint={'every_n_epoch': 1}, tensorboard=None, loss='CrossEntropyLoss', optimizer='Adam', lr_scheduler_type='linear', warmup_ratio=0.0, warmup_steps=0, device_str=None, n_gpu=-1, sync_bn=False, freeze_bn=False)
    [epoch 1/2] loss=2.402, metric=0.0, eval_loss=2.254, eval_metric=0.0: 100%|██████████| 10/10 [00:32<00:00,  3.25s/step]
    [epoch 2/2] loss=1.88, metric=0.1, eval_loss=1.855, eval_metric=0.1: 100%|██████████| 10/10 [00:22<00:00,  1.14step/s]  