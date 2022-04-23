---
id: quick-start
title: Quick Start
---


Follow steps below to get started with a jupyter notebook for how to train a Towhee operator. This example fine-tunes a pretrained model (eg. resnet-18) with a fake dataset.

# 1. Download Operator


```python
! git clone https://towhee.io/towhee/resnet-image-embedding.git
%cd resnet-image-embedding
%ls
```

    Cloning into 'resnet-image-embedding'...
    remote: Enumerating objects: 220, done.[K
    remote: Counting objects: 100% (220/220), done.[K
    remote: Compressing objects: 100% (212/212), done.[K
    remote: Total 220 (delta 119), reused 0 (delta 0), pack-reused 0[K
    Receiving objects: 100% (220/220), 908.00 KiB | 279.00 KiB/s, done.
    Resolving deltas: 100% (119/119), done.
    /media/supermicro/DATA1/zhangchen_workspace/towhee_examples/resnet-image-embedding
    [0m[01;34mexamples[0m/                     requirements.txt
    [01;35mILSVRC2012_val_00049771.JPEG[0m  resnet_image_embedding.py
    __init__.py                   resnet_image_embedding.yaml
    README.md                     resnet_training_yaml.yaml


Then run Python scripts in following steps to train and test a Towhee operator.

# 2. Setup Operator

Create operator and load model by name.


```python
# sys.path.append('..')
from resnet_image_embedding import ResnetImageEmbedding
from towhee.trainer.training_config import TrainingConfig
from torchvision import transforms
from towhee import dataset
op = ResnetImageEmbedding('resnet18', num_classes=10)
```

# 3. Configure Trainer:

Modify training configurations on top of default values.


```python
# build a training config:
training_config = TrainingConfig(
    batch_size=2,
    epoch_num=2,
    output_dir='quick_start_output'
)
```

# 4. Prepare Dataset

The example here uses a fake dataset for both training and evaluation.


```python
# prepare the dataset
fake_transform = transforms.Compose([transforms.ToTensor()])
train_data = dataset('fake', size=20, transform=fake_transform)
eval_data = dataset('fake', size=10, transform=fake_transform)
```

# 5. Start Training


Now everything is ready, start training.




```python
op.train(
    training_config,
    train_dataset=train_data,
    eval_dataset=eval_data
)
```

    2022-03-23 12:37:45,355 - 139993194329920 - trainer.py-trainer:319 - WARNING: TrainingConfig(output_dir='quick_start_output', overwrite_output_dir=True, eval_strategy='epoch', eval_steps=None, batch_size=2, val_batch_size=-1, seed=42, epoch_num=2, dataloader_pin_memory=True, dataloader_drop_last=True, dataloader_num_workers=0, lr=5e-05, metric='Accuracy', print_steps=None, load_best_model_at_end=False, early_stopping={'monitor': 'eval_epoch_metric', 'patience': 4, 'mode': 'max'}, model_checkpoint={'every_n_epoch': 1}, tensorboard={'log_dir': None, 'comment': ''}, loss='CrossEntropyLoss', optimizer='Adam', lr_scheduler_type='linear', warmup_ratio=0.0, warmup_steps=0, device_str=None, sync_bn=False, freeze_bn=False)
    [epoch 1/2] loss=2.654, metric=0.15, eval_loss=2.445, eval_metric=0.2: 100%|████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  6.03step/s]
    [epoch 2/2] loss=1.908, metric=0.2, eval_loss=1.826, eval_metric=0.2: 100%|█████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 35.52step/s]

With a successful training, you will see progress bar below and a `quick_start_output` folder containing training results.
