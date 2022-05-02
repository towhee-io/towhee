---
id: image-embedding
title: Image embedding pipelines
---

Image embedding pipelines are used for reduction the dimensionality of the input image processed by general neural networks, including deep neural networks and transformers. They are important for images’ classification related applications, because these correspond to a huge dimension data (e.g. a 20 megapixel camera picture with 3 RGB layers means 60 millions of integers as the total info stored in the image).

### Popular Scenarios

- [Reverse image search](/03-Tutorials/reverse-image-search.md)
- [Image deduplication](/03-Tutorials/image-deduplication.md)
- Copyright infringement detection
- Item tagging
- Celebrity tagging

### Pipelines

These are the pipelines containing pre-trained ResNet models implemented based on this [paper](https://arxiv.org/pdf/1512.03385.pdf):

**[image-embedding-resnet50](https://hub.towhee.io/towhee/image-embedding-resnet50)**

**[image-embedding-resnet101](https://hub.towhee.io/towhee/image-embedding-resnet101)**

These are the pipelines containing pre-trained EfficientNet models implemented based on this [paper](https://arxiv.org/pdf/1905.11946.pdf):

**[image-embedding-efficientnetb5](https://hub.towhee.io/towhee/image-embedding-efficientnetb5)**

**[image-embedding-efficientnetb7](https://hub.towhee.io/towhee/image-embedding-efficientnetb7)**

These are the pipelines containing pre-trained ViT (vision transformer) implemented based on this [paper](https://arxiv.org/pdf/2010.11929.pdf):

**[image-embedding-vitlarge](https://hub.towhee.io/towhee/image-embedding-vitlarge)**

These are the pipelines containing pre-trained Swin Transformers implemented based on this [paper](https://arxiv.org/pdf/2103.14030v1.pdf):

**[image-embedding-swinbase](https://hub.towhee.io/towhee/image-embedding-swinbase)**

**[image-embedding-swinlarge](https://hub.towhee.io/towhee/image-embedding-swinlarge)**

Each pipeline mentioned above employs a single model. In contrast, an ensemble combines multiple models to generate a better embedding, at the cost of higher resource consumption. Towhee offers image embedding pipelines based on emsembles:

**[image-embedding-efficientnetb7-swinlarge-ensemble](https://hub.towhee.io/towhee/image-embedding-efficientnetb7-swinlarge-ensemble)**

### Pipeline family

All the pipelines mentioned above belong to `image-embedding-pipeline family`. Pipeline members of the family follow [this interface](https://hub.towhee.io/towhee/image-embedding-pipeline-template).
