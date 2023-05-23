# Flat Map

## Introduction

A flat map node flattens the results after applying the function to every row of input, and returns the flattened data respectively.

The returned data can have the same count or more number of rows compared with the input. This is one of the major differences between `flat_map` and `map`, where `map` always returns the same number of rows as input. Refer to [flat_map API](/04-API%20Reference/01-Pipeline%20API/04-flat-map.md) for more details.

The figure below illustrates how `flat_map` applies the transformation to each row of inputs.

![img](https://github.com/towhee-io/data/blob/main/image/docs/flat_map_intro.png?raw=true)

Note that a list is returned by the flat node after it applies a function. Then the flat map node flattens the list.



## Example

We use the `flat_map(input_schema, output_schema, fn, config=None)` interface to create a flat map node. Note that the input of the `fn` function should follow the `input_schema` while the output of the `fn` function should follow the `output_schema`.



Now let's take an object detection and image feature extraction pipeline as an example to demonstrate how to use a flat map node.



The pipeline, `obj_embedding`,detects objects in images, crops images, and then extracts the feature embeddings of the target objects in the images.

```Python
from towhee import pipe, ops, DataCollection

obj_embedding = (
    pipe.input('url')
        .map('url', 'img', ops.image_decode.cv2_rgb())
        .flat_map('img', ('box', 'class', 'score'), ops.object_detection.yolo())
        .flat_map(('img', 'box'), 'object', ops.towhee.image_crop())
        .map('object', 'embedding', ops.image_embedding.timm(model_name='resnet50'))
        .output('url', 'object', 'class', 'score', 'embedding')
    )


data = 'https://towhee.io/object-detection/yolo/raw/branch/main/objects.png'
res = obj_embedding(data)
res.size # return 2
```

The DAG of the `obj_embedding` pipeline is illustrated below. Texts on the arrows in the image describes how data is transformed by each node. Outputs of a node is highlighted.

![img](https://github.com/towhee-io/data/blob/main/image/docs/flat_map_example_1.png?raw=true)

The data processing workflow of the main nodes is as follows.

1. **Map:** Uses the [image-decode/cv2-rgb](https://towhee.io/image-decode/cv2-rgb) operator to decode image URLs (`url`) into images (`img`).
2. **Flat map:** Uses the [object-detection/yolo](https://towhee.io/object-detection/yolo) operator to extract the target object in images (`img`), obtain the positions (`box`) of the target objects in the images, classify information (`class`), and score the results (`score`).
3. **Flat map:** Uses the [towhee/image-crop](https://towhee.io/towhee/image-crop) operator to crop images (`img`) according to the object position (`box`) and obtain and images of the target objects (`object`).
4. **Map:** Uses the  [image_embedding/timm](https://towhee.io/image-embedding/timm) operator to extract the features of each object and obtain the corresponding vector embedding (`embedding`) of the object (`object`).

When the pipeline is running, data transformation in each node is illustrated below.

> Note:
>
> - The data in the `img` and `object` columns are in the format of `towhee.types.Image`. These data represent the decoded images. For easier understanding, these data are displayed as images in the following figure.
> - The data in the `box` column are in the format of `list`. These data are coordinates that represent the location of the objects in the images (Eg. `[448,153,663,375]`). For easier understanding, these data are displayed as the dotted squares in the images.

![img](https://github.com/towhee-io/data/blob/main/image/docs/map_example_2.png?raw=true)

There are two flat map nodes in the pipeline. 

- **flat_map('img', ('box', 'class', 'score'), ops.object_detection.yolo())**

This node applies [object-detection/yolo](https://towhee.io/object-detection/yolo) operator to `img`. Note that there are two objects in the example image. The operator returns a list containing two tuples with three fields (`box`， `class`,  and `score`). Then these tuples are flattened into two rows as output.

Note that `url` 和 `img` are scalars, and the values are automatically repeated to align with the row length (2).

- **flat_map(('img', 'box'), 'object', ops.towhee.image_crop())**

This node applies [towhee/image-crop](https://towhee.io/towhee/image-crop) operator to `img` to crop the detected objects based on the boundary boxes `box`. The operator returns a list of two images that are then flattened.



## Notes

- A flat map node returns a list after applying a function to the input.
