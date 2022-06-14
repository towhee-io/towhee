---
id: train-a-bird-classification-model
title: Train a Bird Classification Model
---

Follow steps below to get started with a jupyter notebook for how to train a Towhee operator. This example fine-tunes a pretrained ResNet model (eg. resnet-34 pretrained by ImageNet) with a bird dataset.
# 1. Download Operator
Download operator files together with the jupyter notebook.


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
    Receiving objects: 100% (220/220), 908.00 KiB | 12.00 KiB/s, done.
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
# import sys
# sys.path.append('..')
from resnet_image_embedding import ResnetImageEmbedding

# Set num_classes=400 for ResNet34 model (400 classes of birds in total)
op = ResnetImageEmbedding('resnet34', num_classes=400)
```

# 3. Configure Trainer:
Modify training configurations on top of default values.


```python
from towhee.trainer.training_config import TrainingConfig

training_config = TrainingConfig()
training_config.batch_size = 32
training_config.epoch_num = 2
training_config.output_dir = 'bird_output'
```

# 4. Prepare Dataset
Download [BIRDS 400](https://www.kaggle.com/gpiosenka/100-bird-species) from Kaggle Dataset. And then create the dataset with local path and transform.



```python
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder

bird_400_path = '/home/zhangchen/zhangchen_workspace/dataset/bird_400'
# bird_400_path = '/path/to/your/dataset/bird_400/'

std = (0.229, 0.224, 0.229)
mean = (0.485, 0.456, 0.406)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std),
                                transforms.RandomHorizontalFlip(p=0.5)
                               ])
train_data = ImageFolder(os.path.join(bird_400_path, 'train'), transform=transform)
eval_data = ImageFolder(os.path.join(bird_400_path, 'valid'), transform=transform)
```

# Start Training
Now start training the operator with Bird-400 dataset.


```python
op.train(training_config, train_dataset=train_data, eval_dataset=eval_data)
```

    2022-03-29 15:10:39,214 - 139982696838976 - trainer.py-trainer:319 - WARNING: TrainingConfig(output_dir='bird_output', overwrite_output_dir=True, eval_strategy='epoch', eval_steps=None, batch_size=32, val_batch_size=-1, seed=42, epoch_num=2, dataloader_pin_memory=True, dataloader_drop_last=True, dataloader_num_workers=0, lr=5e-05, metric='Accuracy', print_steps=None, load_best_model_at_end=False, early_stopping={'monitor': 'eval_epoch_metric', 'patience': 4, 'mode': 'max'}, model_checkpoint={'every_n_epoch': 1}, tensorboard={'log_dir': None, 'comment': ''}, loss='CrossEntropyLoss', optimizer='Adam', lr_scheduler_type='linear', warmup_ratio=0.0, warmup_steps=0, device_str=None, sync_bn=False, freeze_bn=False)
    [epoch 1/2] loss=2.181, metric=0.672, eval_loss=2.157, eval_metric=0.947: 100%|█████████████████████████████████████████████████| 1824/1824 [03:53<00:00,  7.82step/s]
    [epoch 2/2] loss=0.37, metric=0.939, eval_loss=0.391, eval_metric=0.962: 100%|██████████████████████████████████████████████████| 1824/1824 [03:47<00:00,  8.39step/s]

# 6. Predict
With the fine-tuned model, you can then use the operator to classify a bird picture.

```python
from towhee.trainer.utils.plot_utils import predict_image_classification
import random
import matplotlib.pyplot as plt

img_index = random.randint(0, len(eval_data))
img = eval_data[img_index][0]
img_np = img.numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
img_np = img_np * std + mean
plt.axis('off')
plt.imshow(img_np)
plt.show()
img_tensor = eval_data[img_index][0].unsqueeze(0).to(op.trainer.configs.device)

prediction_score, pred_label_idx = predict_image_classification(op.model, img_tensor)
print('It is {}.'.format(eval_data.classes[pred_label_idx].lower()))
print('probability = {}'.format(prediction_score))
```

    2022-03-29 15:18:20,841 - 139982696838976 - image.py-image:725 - WARNING: Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



    
![png](3_train_a_bird_classification_model_files/3_train_a_bird_classification_model_11_1.png)
    


    It is mandrin duck.
    probability = 0.9747885465621948


# 7.Interpret model
If you try to understand why this image will be classified as a mandrin duck, you can use `interpret_image_classification` utils in towhee, which using [captum](https://captum.ai/) as backend. So you must install it first using `pip install captum` or `conda install captum -c pytorch`.

```python
from PIL import Image
import numpy as np
from towhee.trainer.utils.plot_utils import interpret_image_classification

pil_img = Image.fromarray(np.uint8(img_np * 255))
val_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std),
                                    ])
interpret_image_classification(op.model.to('cpu'), pil_img, val_transform, "Occlusion")
interpret_image_classification(op.model.to('cpu'), pil_img, val_transform, "GradientShap")
interpret_image_classification(op.model.to('cpu'), pil_img, val_transform, "Saliency")
```


    
![png](3_train_a_bird_classification_model_files/3_train_a_bird_classification_model_13_0.png)
    



    
![png](3_train_a_bird_classification_model_files/3_train_a_bird_classification_model_13_1.png)
    



    
![png](3_train_a_bird_classification_model_files/3_train_a_bird_classification_model_13_2.png)
    





    (0.9745060205459595, 261)
