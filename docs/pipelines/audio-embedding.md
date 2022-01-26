---
id: audio-embedding
title: Audio embedding pipelines
---

Audio describes human-hearable sound, which is normally recorded and transmitted via sound files in formats such as MP3 or WAV. Audio embedding is the process of converting audio files into vector representations. Here, we list some of our built-in pipelines for generating _music_ embeddings.

### Popular scenarios

- Recognize audio events or scenes
- Tag music for genres, artists, emotion
- Identify music with a short clip
- Music copyright infringement

### Pipelines

Music and other audio snippets have seen great success using 1D convolutional neural networks, just as images have. Most models include some form of preprocessing such as data cropping, downsampling, and other transformations. Towhee maintains the following pipelines for audio generation:

**[audio-embedding-vggish](https://towhee.io/towhee/audio-embedding-vggish)**

This pipeline contains a pre-trained model based on [VGGish](https://arxiv.org/abs/1609.09430). VGGish is a supervised model pretrained with [AudioSet](https://research.google.com/audioset/), a large scale audio classification task.

**[audio-embedding-clmr](https://towhee.io/towhee/audio-embedding-clmr)**

The pipeline contains a pre-trained model based on [CLMR](https://arxiv.org/abs/2103.09410). CLMR is a semi-supervised encoder-based model which works well for audio/music fingerprinting.

More on the way ...
