# Built-in Layers

Towhee offers a large group of neural network building blocks which can be used as "legos" for fast construction of machine learning models. Classical, modern, and hypermodern blocks are all supported.

Layer blocks, referred to as `layers` in Towhee, are reusable and time-saving, thus not needing to be reimplemented for developers, researchers, and machine learning projects. For computation concerns, layers can be optimized for different platforms. See Towhee's layers in [repo](https://github.com/towhee-io/towhee/tree/main/towhee/models/layers). Developers are welcome to contribute to the diversity of reusable layers.

**Example**

[Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf) proposes a hierarchical transformer whose representation is computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size[1], which is trained on [imagenet dataset](https://image-net.org/download.php)

Such a network takes only **71** lines using Towhee's Layer subframework instead of **476** lines.

[PatchEmbed2D](https://github.com/towhee-io/towhee/blob/main/towhee/models/swin_transformer/patch_embed2d.py), [PatchMerging](https://github.com/towhee-io/towhee/blob/main/towhee/models/swin_transformer/patch_merging.py), and [SwinTransformerBlock](https://github.com/towhee-io/towhee/blob/main/towhee/models/swin_transformer/swin_transformer_block.py) are used to create a Swin Transformer modal.

**Architecture of Swin Transformer**

![img](./swin_arch.png)
