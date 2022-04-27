---
id: audio-embedding
title: Audio embedding pipelines
---

Audio is defined as any human-hearable sound; audio embedding is the process of converting audio files (`mp3`, `wav`, etc...) into vector representations. Here, we list some of our built-in pipelines for generating audio embeddings.

### Popular scenarios

- [Identify music with an audio snippet](/Tutorials/music-recognition-system.md)
- Recognize audio events or scenes
- Tag music for genres, artists, emotion
- Music copyright infringement

### Pipelines

Audio tasks have seen incredible strides using 1-dimensional convolutional neural networks. Just as with CNNs used for image embedding, most audio embedding models  include some form of preprocessing such as data cropping and downsampling. Towhee maintains the following audio embedding pipelines:

**[audio-embedding-vggish](https://towhee.io/towhee/audio-embedding-vggish)**

This pipeline contains a pre-trained model based on [VGGish](https://arxiv.org/abs/1609.09430). VGGish is a supervised model trained using the [AudioSet](https://research.google.com/audioset/) dataset, a large scale audio classification task.

**[audio-embedding-clmr](https://towhee.io/towhee/audio-embedding-clmr)**

This pipeline contains a pre-trained model based on [CLMR](https://arxiv.org/abs/2103.09410), also known as _Contrastive Learning of Musical Representations_. CLMR is a semi-supervised encoder-based model which works well for music fingerprinting. Its performance on generic audio clips is untested.
