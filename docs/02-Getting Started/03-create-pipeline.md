# Create Your First Pipeline

A pipeline connects multiple function nodes to provide a full set of data processing capabilities. Towhee provides various built-in nodes for general applications. But you can also defines nodes according to your own needs. 



This tutorial walks you through how to create a pipeline and provides some examples. 

> Refer to [Node Types](../03-User Guides/01-Pipeline Programing Guide/01-node-types.md) for more information about nodes.

## Create a simple pipeline 

The following example demonstrates how create a simple pipeline with a Lambda function.

```Python
from towhee import pipe

add_one = (
    pipe.input('x')
        .map('x', 'y', lambda x: x + 1)
        .output('y')
)

res = add_one(0).get()
```

In the above example, we define a pipeline `add_one` which increases the value of the inputs by 1 and returns the results. This pipeline contains three nodes.

- **input('x')**

  The input node defines the pipeline's input schema which contains a single variable `'x'`.

- **map('x', 'y', lambda x: x+1)**
  
  The `map` node applies the lambda `x: x + 1` to each of the input value.
  
- **output('y')**
  
  The output node defines the pipeline's output schema and ends the pipeline definition. Once called, a pipeline instance is created and returned.

In line 9, we pass an input value `0` to the pipeline, and will get an output value `1`. 



Of course, you can also use a function in a pipeline definition as illustrate below:

```Python
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



## Use Towhee Operator to create a more complicated pipeline

In real-world applications, a pipeline requires nodes to have much more complicated functions than the above example. Towhee provides the following three types of built-in operators to help you quickly create a more complicated pipeline. 

- **Data processing operators** such as image transformations, image/audio/video decode, tokenizers, vector normalization, etc.
- **Neural network models or libraries** such as CLIP4Clip pretrained model, HuggingFace model adapter, OpenAI embedding API wrapper, etc. 
- **Connectors** such as Milvus connector, HBase connector, etc.

The following example demonstrates how to use a TIMM (Torch Image Model) model to generate image embeddings.

> When the pipeline runs for the first time, it may take several minutes to download operators and models.

```Python
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

This pipeline contains four nodes.

- **input('url')**
  
  The input node defines the pipeline's input schema which contains a single variable `url`.
  
- **map('url', 'img', ops.image_decode.cv2())**

  This map node takes each `url` as an input. This node fetches the image and uses OpenCV for image decoding. The output `img` is the decoded image data.

- **map('img', 'embedding', ops.image_embedding.timm(model_name='resnet50'))**

  This map node takes each `img` as an input. This node uses the `resnet50` model from TIMM to generate the feature vector. The output `embedding` is the embedding vector for each of the input image.

- **output('embedding')**

  The output node defines the pipeline's output schema and ends the pipeline definition. 
