# Towhee v0.4.0 Release Notes

#### Highlights
- The Towhee website has a new look and feel! The new Towhee website includes several important docs, including in-depth Towhee tutorials, pipeline and operator summaries, development and contributing guides, and more. See https://docs.towhee.io. If you have any feedback for the website design or encounter any bugs, please open an issue through Github.
- Towhee now offers pre-built embedding pipelines that use transformer-based models: SwinTransformer and ViT.
- Towhee now offers pipelines with ensemble embedding models.
- Official support for several different operating systems (Ubuntu 18.04, Ubuntu 20.04, macOS 10, macOS 11 Intel CPU, and Windows 10) is here!
- A major refactor was completed - this refactor uses a new task scheduling framework and allows Operators to be based on generators, added several new iterators over input DataFrames (flatmap, concat, etc), added runners for operators, and more. See below for a full set of details.
- Towhee now supports concat operators in pipeline.
- Towhee now supports local caching - previously downloaded pipelines and operators are now cached within the ~/.towhee directory for future use.

#### Pipelines
- Image embedding pipeline(s) based on Resnet models (paper):
  - image-embedding-resnet50
  - image-embedding-resnet101
- Image embedding pipeline(s) based on EfficientNet models (paper):
  - image-embedding-efficientnetb5
  - image-embedding-efficientnetb7
- Image embedding pipeline(s) based on ViT (paper):
  - image-embedding-vitlarge
- Image embedding pipeline(s) based on Swin Transformers (paper):
  - image-embedding-swinbase
  - image-embedding-swinlarge
- Image embedding pipeline(s) based on multiple ensemble models:
  - image-embedding-efficientnetb7-swinlarge-concat
  - image-embedding-3ways-ensemble-large-v1
- Music embedding pipeline(s) based on VGGish (paper):
  - audio-embedding-vggish
- Music embedding pipeline(s) based on CLMR (paper):
  - audio-embedding-clmr

#### Operators
- Operator(s) based on Resnet models (paper):
  - resnet-image-embedding
- Operators(s) based on EfficientNet models (paper):
  - efficientnet-image-embedding
- Operators(s) based on ViT (paper):
  - vit-image-embedding
- Operators(s) based on Swin Transformers (paper):
  - swintransformer-image-embedding
- Operators(s) based on VGGish (paper):
  - tf-vggish-audioset
- Operators(s) based on CLMR (paper):
  - clmr-magnatagatune

#### Towhee Framework
- A major refactor was completed - this refactor uses a new task scheduling framework, it also allows Operators to be based on generators, added several new iterators over input DataFrames (flatmap, concat, etc), added runners for operators, and more. See below for a full set of details.
- Engine refactor: #272 #303 #324
- Support concat #296 #339 #359
- Hub pipeline/operator repo download by branch #286
- Local cache for pipeline/operator repo  #297
- Adjust pipeline/operator repo file structure, add hub tools for pipeline/operator project init #315 #323
- Base classes for operators: NNOperator, PyOperator #333
- Clean up towhee third party dependency. Migrate requirements from towhee to operators. #308
- Allow instantiate pipelines by yaml description files. #320

#### Documentation
New pages
- ["Overview"](https://docs.towhee.io/)
- ["Getting started / Quick start"](https://docs.towhee.io/get-started/quick-start)
- ["Getting started / Installation"](https://docs.towhee.io/get-started/install)
- "Tutorials / Reverse image search"
- "Tutorials / Image deduplication"
- "Tutorials / Music recognition"
- "Supported pipelines / Image embedding pipelines"
- ["Supported pipelines / Music embedding pipelines"](https://docs.towhee.io/pipelines/music-embedding)
- ["Developer guides / Contributing / Contributing guide"](https://docs.towhee.io/developer-guides/contributing/contributing-guide)

#### Thanks
Many thanks to all those who contributed to this release!
@binbinlv @Chiiizzzy @derekdqc @filip-halt @fzliu @GuoRentong @guoxiangzhou @jaelgu @jeffoverflow @jennyli-z @junjiejiangjjj @LoveEachDay @NbnbZero @oneseer @shanghaikid @shiyu22 @Tumao727 @wxywb @yanliang567 @zc277584121
