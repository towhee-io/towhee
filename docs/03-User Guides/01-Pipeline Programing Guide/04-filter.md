# Filter

## Introduction

A filter node filters rows based on the return values (T/F) of a given function `func` and takes `filter_columns` as input. The transformation only takes effect on the columns specified by the `input_schema`. Refer to [filter API](/05-API%20Reference/01-Pipeline%20API/05-filter.md) for more details.

![img](https://github.com/towhee-io/data/blob/main/image/docs/filter_intro.png?raw=true)

Since the filter node only filters the input data, the input and output of a filter node should be the same. The function in the filter node takes `filter_columns` as input, which can overlap with one or more specified columns in the `input_schema`.



## Example

We use the `filter(input_schema, output_schema, filter_columns, fn, config=None)` interface to create a filter node.



Now let's take an object detection pipeline as an example to demonstrate how to use a filter node. This pipeline detects objects in images, filters out those unidentified objects, and then extracts the feature embedding of each detected object.

```Python
from towhee import pipe, ops

obj_filter_embedding = (
    pipe.input('url')
        .map('url', 'img', ops.image_decode.cv2_rgb())
        .map('img', 'obj_res', ops.object_detection.yolo())
        .filter(('img', 'obj_res'), ('img', 'obj_res'), 'obj_res', lambda x: len(x) > 0)
        .flat_map('obj_res', ('box', 'class', 'score'), lambda x: x)
        .flat_map(('img', 'box'), 'object', ops.towhee.image_crop())
        .map('object', 'embedding', ops.image_embedding.timm(model_name='resnet50'))
        .output('url', 'object', 'class', 'score', 'embedding')
    )
    
data = ['https://towhee.io/object-detection/yolo/raw/branch/main/objects.png', 'https://github.com/towhee-io/towhee/raw/main/assets/towhee_logo_square.png']
res = obj_filter_embedding.batch(data)
```

The DAG of the `obj_filter_emnedding` pipeline is illustrated below. 

![img](https://github.com/towhee-io/data/blob/main/image/docs/filter_example_1.png?raw=true)

The data processing workflow of the main nodes is as follows.

1. **Map:** Uses the [image-decode/cv2-rgb](https://towhee.io/image-decode/cv2-rgb) operator to decode image URLs (`url`) into images (`img`).
2. **Map:** Uses the [object-detection/yolo](https://towhee.io/object-detection/yolo)  operator to extract target objects in images  (`img`), and returns a list of the information of the target objects (`obj_res`) in the images.
3. **Filter:** Applies the function `lambda x: len(x) > 0` to determine if the value of `obj_res`is greater than 0. In other words, this process filters out any undetected objects in the images.
4. **Flat map:** Applies the `lambda x: x` function to flatten the `obj_res` of objects that meet the filtering condition. The output of this node is three columns - position of the detected objects (`box`), object classification (`class`), and the score of the objects (`score`).
5. **Flat map:** Uses the [towhee/image-crop](https://towhee.io/towhee/image-crop) operator to crop images (`img`) and obtain and images of the target objects (`object`).
6. **Map:** Uses the  [image-embedding/timm](https://towhee.io/image-embedding/timm) operator to extract the features of each object and obtain the corresponding feature embedding (`embedding`)of the objects (`object`) .

When the pipeline is running, data transformation in each node is illustrated below.

> Note:
>
> - The data in the `img` and `object` columns are in the format of `towhee.types.Image`. These data represent the decoded images. For easier understanding, these data are displayed as images in the following figure.
> - The data in the `box` column are in the format of `list`. These data are coordinates that represent the location of the objects in the images (Eg. `[448,153,663,375]`). For easier understanding, these data are displayed as the dotted squares in the images.

![img](https://github.com/towhee-io/data/blob/main/image/docs/filter_example_2.png?raw=true)



## Notes

- A filter node returns a list of values as either `True` or `False` after applying the function. 
- The `input_schema` and the `output_schema` in a filter node should have the same number of columns.
