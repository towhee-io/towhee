<!---
Copyright 2021 Zilliz. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
-->

![https://towhee.io](../../towhee_logo.png#gh-light-mode-only)
![https://towhee.io](../../towhee_logo_dark.png#gh-dark-mode-only)

<h3 align="center">
  <p style="text-align: center;"> <span style="font-weight: bold; font: Arial, sans-serif;">x</span>2vec, Towhee is all you need! </p>
</h3>

<h4 align="center">
    <p>
        <a href="https://github.com/towhee-io/towhee/tree/main/towhee/models">English</a> |
        <b>中文</b> 
    <p>
</h4>

# Towhee 模型

Towhee 在持续跟进和收录各种大受欢迎以及前沿的模型，
通过`towhee.models`([python package](https://pypi.org/project/towhee.models/)) 实现这些复杂的模型，并为用户提供更加简单的调用方式。
下面是一些用法示例：
```python
from towhee.models import vit

# Create model
vit_model = vit.create_model(**kwargs)

# Load pretrained model
pretrained_model = vit.create_model(model_name='vit_base_16x224', pretrained=True)
```
[Towhee Hub](https://towhee.io/tasks/operator) 拥有模型在应用场景或不同任务中的用法，包含了这里的大部分模型。

## 模型列表
| 模型 | 论文 | 任务 | Towhee Hub | 
| --- | --- | --- | --- |
| AcarNet | [Actor-Context-Actor Relation Network for Spatio-temporal Action Localization](https://arxiv.org/pdf/2006.07976.pdf) | action detection | |
| ActionClip | [ActionCLIP: A New Paradigm for Video Action Recognition](https://arxiv.org/pdf/2109.08472v1.pdf) | action classification | [action_classification.actionclip](https://towhee.io/action-classification/actionclip) |
| All in one| [All in One: Exploring Unified Video-Language Pre-training](https://arxiv.org/pdf/2203.07303v1.pdf) | video question answering | |
| BridgeFormer | [Bridging Video-text Retrieval with Multiple Choice Questions](http://openaccess.thecvf.com//content/CVPR2022/papers/Ge_Bridging_Video-Text_Retrieval_With_Multiple_Choice_Questions_CVPR_2022_paper.pdf) | text-video retrieval | [video_text_embedding.bridge_former](https://towhee.io/video-text-embedding/bridge-former) |
| CLIP | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020v1) | text-image retrieval | [image_text_embedding.clip](https://towhee.io/image-text-embedding/clip) |
| CLIP4CLIP | [CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval](https://arxiv.org/pdf/2104.08860v2.pdf) | text-video retrieval | [video_text_embedding.clip4clip](https://towhee.io/video-text-embedding/clip4clip) |
| Collaborative Experts | [Use What You Have: Video Retrieval Using Representations From Collaborative Experts](https://arxiv.org/pdf/1907.13487) | text-video retrieval | [video_text_embedding.collaborative_experts](https://towhee.io/video-text-embedding/collaborative-experts) |
| DRL | [Disentangled Representation Learning for Text-Video Retrieval](https://arxiv.org/pdf/2203.07111) | text-video retrieval | [video_text_embedding.drl](https://towhee.io/video-text-embedding/drl) |
| Frozen in Time | [Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval](http://openaccess.thecvf.com//content/ICCV2021/papers/Bain_Frozen_in_Time_A_Joint_Video_and_Image_Encoder_for_ICCV_2021_paper.pdf) | video-text retrieval | [video_text_embedding.frozen_in_time](https://towhee.io/video-text-embedding/frozen-in-time) |
| LightningDot | [LightningDOT: Pre-training Visual-Semantic Embeddings for Real-Time Image-Text Retrieval](https://aclanthology.org/2021.naacl-main.77.pdf) | text-image retrieval | [image_text_embedding.lightningdot](https://towhee.io/image-text-embedding/lightningdot) |
| MDMMT | [MDMMT: Multidomain Multimodal Transformer for Video Retrieval](https://arxiv.org/pdf/2103.10699v1.pdf) | video-text retrieval | [video_text_embedding.mdmmt](https://towhee.io/video-text-embedding/mdmmt) |
| MoViNet | [MoViNets: Mobile Video Networks for Efficient Video Recognition](http://openaccess.thecvf.com//content/CVPR2021/papers/Kondratyuk_MoViNets_Mobile_Video_Networks_for_Efficient_Video_Recognition_CVPR_2021_paper.pdf) | action classification | [action_classification.movinet](https://towhee.io/action-classification/movinet) |
| MPViT | [MPViT: Multi-Path Vision Transformer for Dense Prediction](http://openaccess.thecvf.com//content/CVPR2022/papers/Lee_MPViT_Multi-Path_Vision_Transformer_for_Dense_Prediction_CVPR_2022_paper.pdf) | image classification | [image_embedding.mpvit](https://towhee.io/image-embedding/mpvit) |
| MViT | [Multiscale Vision Transformers]([http://openaccess.thecvf.com//content/ICCV2021/papers/Fan_Multiscale_Vision_Transformers_ICCV_2021_paper.pdf]) | image classification, action classification | [image_embedding.timm](https://towhee.io/image-embedding/timm) [action_classification.pytorchvideo](https://towhee.io/action-classification/pytorchvideo) |
| NNFp | [Neural Audio Fingerprint for High-specific Audio Retrieval based on Contrastive Learning](https://arxiv.org/pdf/2010.11910) | audio fingerprint | [audio_embedding.nnfp](https://towhee.io/audio-embedding/nnfp) |
| Omnivore | [Omnivore: A Single Model for Many Visual Modalities](http://openaccess.thecvf.com//content/CVPR2022/papers/Girdhar_Omnivore_A_Single_Model_for_Many_Visual_Modalities_CVPR_2022_paper.pdf) | action classification | [action_classification.omnivore](https://towhee.io/action-classification/omnivore) |
| Perceiver | [Perceiver: General Perception with Iterative Attention](https://arxiv.org/pdf/2103.03206v2.pdf) | image classification, audio classification | |
| RepMLP | [RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality](https://arxiv.org/pdf/2112.11081) | image classification | |
| RetinaFace | [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/pdf/1905.00641v2.pdf) | face detection | [face_detection.retinaface](https://towhee.io/face-detection/retinaface) |
| Swin Transformer | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](http://openaccess.thecvf.com//content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf) | image classification | [image_embedding.timm](https://towhee.io/image-embedding/timm) |
| TSM | [TSM: Temporal Shift Module for Efficient Video Understanding](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.pdf) | action classification | [action_classification.tsm](https://towhee.io/action-classification/tsm) |
| TransRAC | [TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting](https://arxiv.org/pdf/2204.01018) | repetitive action count | |
| UniFormer | [UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning](https://arxiv.org/pdf/2201.04676v3.pdf) | action classification | [action_classification.uniformer](https://towhee.io/action-classification/uniformer) |
| VGGish | [CNN Architectures for Large-Scale Audio Classification](https://arxiv.org/pdf/1609.09430v2.pdf) | audio embedding | [audio_embedding.vggish](https://towhee.io/audio-embedding/vggish) |
| Video Swin Transformer | [Video Swin Transformer](http://openaccess.thecvf.com//content/CVPR2022/papers/Liu_Video_Swin_Transformer_CVPR_2022_paper.pdf) | action classification | [action_classification.video_swin_transformer](https://towhee.io/action-classification/video-swin-transformer) |
| Violet | [VIOLET: End-to-End Video-Language Transformers with Masked Visual-token Modeling](https://arxiv.org/pdf/2111.12681v2.pdf) | video question answering | |
| ViT | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/pdf?id=YicbFdNTTy) | image classification | [image_embedding.timm](https://towhee.io/image-embedding/timm) |
| Wave-ViT | [Wave-ViT: Unifying Wavelet and Transformers for Visual Representation Learning](http://arxiv.org/pdf/2207.04978) | image recognition, object detection, instance segmentation | |



