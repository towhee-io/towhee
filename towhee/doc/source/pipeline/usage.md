# Pipeline
A pipeline is composed of several operators interconnected in the form of a DAG (directed acyclic graph). This DAG can direct complex functionalities, such as embedding feature extraction, data tagging, and cross modal data analysis.

## A Simple Pipeline 

```python
from towhee import pipe

add_one = (
    pipe.input('x')
        .map('x', 'y', lambda x: x + 1)
        .output('y')
)

res = add_one(0).get()
```
Above we defined an pipeline `add_one`, whose function is to add one to the input number and return the result. It contains three nodes

1. input('x')
This node defines the pipeline's input schema，which contains a single variable 'x'.

2. map('x', 'y', lambda x: x+1)
The map node will apply the lambda x: x + 1 to each of the input value.

3. output('y')
  This node defines the pipeline's output schema. The output ends the pipeline definition. Once called, a pipeline instance will be created and returned.
In line 9, we pass an input value 0 to the pipeline, and will get a return value 1. 

Of cause, you can also use a function in a pipeline definition, as illustrate below:

```python
from towhee import pipe

def my_func(x):
   return x + 1

add_one = (
    pipe.input('x')
        .map('x', 'y', my_func)
        .output('y')
)

res = add_one(0).get()
```
## Use Towhee Operator
In actual scenarios, the nodes used in pipelines are much more complex than the above example. Towhee provides three classes of built-in operators to help users quickly build pipelines.

1. The first class is data processing operators, such as image transformations, image/audio/video decoding, tokenizers, vector normalization, etc.

2. The second class is neural network models or model libraries, such as Clip4CLIP pretrained models, Huggingface model adapters, OpenAI embedding API wrappers, etc.

3. The third class is connectors, such as Milvus connectors, HBase connectors, etc.

Through these built-in nodes, users can easily build end-to-end AI pipelines without having to implement various algorithms and models from scratch. Towhee aims to provide users with a simple, efficient, and extensible AI pipeline construction and deployment platform. 


The following example demonstrates using a TIMM (Torch Image Model) model to generate image embeddings:
The pipeline will download the operator and model for the first run, which may take a few minutes. 
```python
from towhee import pipe, ops

img_embedding = (
    pipe.input('url')
        .map('url', 'img', ops.image_decode.cv2())
        .map('img', 'embedding', ops.image_embedding.timm(model_name='resnet50'))
        .output('embedding')
)

url = 'https://github.com/towhee-io/towhee/raw/main/towhee_logo.png'
res = img_embedding(url).get()
```
This pipeline contains four nodes:

1. input('url')
  This node defines the pipeline's input schema，which contains a single variable url.

2. map('url', 'img', ops.image_decode.cv2())
This node takes each url as input. It will fetch the image, and use OpenCV for image decoding. The output img is the decoded image data.

3. map('img', 'embedding', ops.image_embedding.timm(model_name='resnet50'))
This node takes each img as input. It uses the resnet50 model from TIMM to generate the feature vector. The output embedding is the embedding vector for each of the input image.

4. output('embedding')

This node defines the pipeline's output schema.

## pre-defined pipelines
Towhee also provides some predefined pipelines that users can load through the [AutoPipe](https://github.com/towhee-io/towhee/blob/main/towhee/runtime/auto_pipes.py) and [AutoConfig](https://github.com/towhee-io/towhee/blob/main/towhee/runtime/auto_config.py) interfaces.

All available pipelines are under [towhee/pipelines](https://github.com/towhee-io/towhee/tree/main/towhee/pipelines).
In the future, we will put the predefined pipelines on the [Towhee Hub](https://towhee.io/).
This is an example of sentence embedding.
```python
from towhee import AutoPipes, AutoConfig

# sentence embedding pipeline
config = AutoConfig.load_config('sentence_embedding')
config.model = 'all-MiniLM-L6-v2'
config.device = 0

embed_pipe = AutoPipes.pipeline('sentence_embedding', config)
print(embed_pipe('How are you?').to_list())
```
