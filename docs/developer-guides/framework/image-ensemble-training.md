---
id: image-ensemble-training
title: Image ensemble training
---

Towhee provides ensemble pipelines for ensemble learning, which utilize multiple models to jointly solve problems like classification, image retrieval and so on. In addition to this, grid search and fusion model training solutions are included in Towhee, allowing developers to spend more time focussing on overall system design.

In the future, a multi-modal fusion will be released to help improve the tools used embedding fusion.

Why would someone use an ensemble? We tested an image embedding ensemble for image retrieval on the Imagenet2012 dataset. When we used one model to extract embedding vectors, we got a best result of (0.6034 mAP@50) out of the 4 models we tested. When we concated the embeddings from the four different models, we got a better result of (0.6161 mAP@50), a 1.3% improvement. This result shows that the embedding ensemble method does improve the retrieval result compared to a single model.

### Imagenet dataset ensemble experiments

**Dataset**: ImageNet(Train:128k val: 50k)

**Evaluation Metric**: mAP@50(ImageNet val dataset has 50 images for each class)

**Evaluation Methodology**: for every image in the ImageNet val dataset, return a sorted list(size 51) for most similar images in the dataset, calculate the metric score for the 50 closest images (first in list ignored as it is the query image). In this test, the images tagged with the same labels as the queried image would be considered positive, and the others would be considered negative.

### Single model experiments

| exp | Models                        | dimension | mAP@50 score       | rank |
| --- | ----------------------------- | --------- | ------------------ | ---- |
| 1   | tf_efficientnet_b7            | 2560      | 0.6034070404139978 | 1    |
| 2   | swin_large_patch4_window7_224 | 1536      | 0.5879429226876567 | 2    |
| 3   | vit_large_patch16_224         | 1024      | 0.5844048334876    | 3    |
| 4   | resnet101                     | 2048      | 0.5490212793579341 | 4    |

### 1 models output concat experiments(grid search, w1, w2 in [1, 2, 3, 4, 5])

| exp | Models | dimension | mAP@50 score | rank |
| --- | --- | --- | --- | --- |
| 1 | tf_efficientnet_b7(5.0)+ swin_large_patch4_window7_224(3.0) | 2560+1536 | 0.6161593062594867 | 1 |
| 2 | tf_efficientnet_b7(3.0)+ swin_large_patch4_window7_224(2.0) | 2560+1536 | 0.615946691446722 | 2 |
| 3 | tf_efficientnet_b7(2.0)+ swin_large_patch4_window7_224(1.0) | 2560+1536 | 0.6155402608223787 | 3 |
| ... | tf_efficientnet_b7(...)+ swin_large_patch4_window7_224(...) | 2560+1536 |  | ... |
| 4 | tf_efficientnet_b7(5.0)+ vit_large_patch16_224(1.0) | 2560+1024 | 0.6093466843189377 | 9 |
| ... |  |  |  | ... |
| 5 | resnet101(2.0)+ tf_efficientnet_b7(1.0) | 2048+2560 | 0.607685290190145 | 11 |
|  |  |  |  |  |

### 3 models output concat experiments(grid search, w1, w2, w3 in [1, 2, 3, 4, 5])

| exp | Models | dimension | mAP@50 score | rank |
| --- | --- | --- | --- | --- |
| 1 | resnet101(5.0)+ tf_efficientnet_b7(3.0)+ swin_large_patch4_window7_224(2.0) | 2048+2560+1536 | 0.6175953128000602 | 1 |
| 2 | resnet101(4.0)+ tf_efficientnet_b7(2.0)+ swin_large_patch4_window7_224(1.0) | 2048+2560+1536 | 0.6171936947440357 | 2 |
| 3 | resnet101(4.0)+ tf_efficientnet_b7(3.0)+ swin_large_patch4_window7_224(2.0) | 2048+2560+1536 | 0.617173806706154 | 3 |
| ... | resnet101(...)+ tf_efficientnet_b7(...)+ swin_large_patch4_window7_224(...) | 2048+2560+1536 |  | ... |
| 4 | tf_efficientnet_b7(5.0)+ swin_large_patch4_window7_224(5.0)+ vit_large_patch16_224tf_efficientnet_b7(1.0) | 2560+1536+1024 | 0.6136430147338187 | 40 |
|  |  |  |  |  |

### 4 models output concat experiments(grid search, w1, w2, w3, w4 in [1, 2, 3, 4, 5])

| exp | Models | dimension | mAP@50 score | rank |
| --- | --- | --- | --- | --- |
| 1 | resnet101(5.0)+ tf_efficientnet_b7(5.0)+ swin_large_patch4_window7_224(5.0)+ vit_large_patch16_224(1.0) | 2048+2560+1536+ 1024 | 0.6140855977227708 | 1 |
| 2 | resnet101(5.0)+ tf_efficientnet_b7(5.0)+ swin_large_patch4_window7_224(4.0)+ vit_large_patch16_224(1.0) | 2048+2560+1536+ 1024 | 0.6140550356453067 | 2 |
| 3 | resnet101(4.0)+ tf_efficientnet_b7(5.0)+ swin_large_patch4_window7_224(5.0)+ vit_large_patch16_224(1.0) | 2048+2560+1536+ 1024 | 0.6139230645898284 | 3 |
| ... |  |  |  | ... |
|  |  |  |  |  |

### History experiments(David)

| Exp | Models | Description | dimension | mAP@50 score |
| --- | --- | --- | --- | --- |
| 1 | Resnet18 | single model | 512 | 0.1970 |
| 2 | Resnet34 | single model | 512 | 0.3194 |
| 3 | Resnet50 | single model | 2048 | 0.4552 |
| 4 | InceptionV3 | single model | 2048 | 0.4180 |
| 5 | InceptionV4 | single model | 1536 | 0.5167 |
| 6 | Resnet18(1.0) + Resnet34(1.0) | Weigted concat and normalization | 1024 | 0.2931 |
| 7 | Resnet18(1.0) + Resnet34(0.5) | Weigted concat and normalization | 1024 | 0.2483 |
| 8 | Resnet34(1.0) + InceptionV3(1.0) | Weigted concat and normalization | 2560 | 0.3820 |
| 9 | Resnet50(1.0) + InceptionV4(1.0) | Weigted concat and normalization | 3584 | 0.5231 |
| 10 | Resnet50(1.5) + InceptionV4(1.0) | Weigted concat and normalization | 3584 | 0.5250 |
| 11 | Resnet101 | single model | 2048 | 0.5490 |
| 12 | Resnet101(10.0) + InceptionV4(1.0) | Weigted concat and normalization | 3584 | 0.5638 |
| 13 | Resnet101 InceptionV4 | Results merge and reranking | / | 0.5341 |
| 14 | tf_efficientnet_b7 + swin_large_patch4_window7_224 | Embedding fixed, finetune fusion layer | 512 | 0.6408 |
| 15 | tf_efficientnet_b7 + vit_large_patch16_224 | Embedding fixed, finetune fusion layer | 512 | 0.5790 |
| 16 | tf_efficientnet_b7 + resnet101 | Embedding fixed, finetune fusion layer | 512 | 0.6256 |
| 17 | tf_on+ resnet101 + swin_large_patch4_window7_224 | Embedding fixed, finetune fusion layer | 512 | 0.6421 |
| 18 | tf_efficientnet_b7+ resnet101 + swin_large_patch4_window7_224 | Concat after project separately | 512 | 0.6419 |
| 19 | tf_efficientnet_b7+ resnet101 + swin_large_patch4_window7_224 | Learing the ratio coefficient for concating | 2048+2560+1536 | 0.6171 |
| 20 | tf_efficientnet_b7+ resnet101 + swin_large_patch4_window7_224 | Embedding fixed, finetune fusion layer | 1024 | 0.6517 |
| 21🌟 | tf_efficientnet_b7+ resnet101 + swin_large_patch4_window7_224 | Embedding fixed, finetune fusion layer | 2048 | 0.6543(SOTA) |
| 22 | tf_efficientnet_b7+ resnet101 + swin_large_patch4_window7_224 + vit_large_patch16_224 | Embedding fixed, finetune fusion layer | 2048 | 0.6382 |
