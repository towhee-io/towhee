# Window All

## Introduction

A window all node batches all input rows into one, and returns the result by applying a function to the window. Refer to [window_all API](../../04-API Reference/01-Pipeline API/08-window-all.md) for more details.

![img](https://github.com/towhee-io/data/blob/main/image/docs/window_all_intro.png?raw=true)

Note that the function in the window all map takes the whole window as input. So, the data type of the input is a list. Each element in the list corresponds to a row.



## Example

We use the `window_all(input_schema, output_schema, fn, config=None)` interface to create a window all node. 



Now let's take an example pipeline to demonstrate how to use a window all node. The example pipeline extracts the feature embedding of each video frame and then merges all the feature embeddings into one ndarray in NumPy.

```Python
from towhee import pipe, ops
import numpy as np

def merge_ndarray(x):
    return np.concatenate(x).reshape(-1, x[0].shape[0])

video_embedding = (
    pipe.input('url')
        .flat_map('url', 'frame', ops.video_decode.ffmpeg())
        .map('frame', 'embedding', ops.image_embedding.timm(model_name='resnet50'))
        .window_all('embedding', 'video_embedding', merge_ndarray)
        .output('url', 'video_embedding')
)

data = 'https://raw.githubusercontent.com/towhee-io/examples/0.7/video/reverse_video_search/tmp/Ou1w86qEr58.gif'
res = video_embedding(data)
```

The DAG of the `video_embedding` pipeline is illustrated below. 

![img](https://github.com/towhee-io/data/blob/main/image/docs/window_all_example_1.png?raw=true)

The data processing workflow of the main nodes is as follows.

1. **Flat map:** Uses the [video-decode/ffmpeg](https://towhee.io/video-decode/ffmpeg) operator to decode video URLs (`url`) into a list of video frames (`List(towhee.types.VideoFrame)`), and then flattens this list into multiple rows. Each row contains one video frame.
2. **Map:** Uses the [image-embedding/timm](https://towhee.io/image-embedding/timm) operator to extract the feature embeddings (`embedding`) of the selected frames.
3. **Window all:** Merges all the feature embeddings (`embedding`) in to one window and then applies a predefined `merge_ndarray` function to turn all data in the window into an ndarray (`video_embeddings`).



When the pipeline is running, data transformation in each node is illustrated below.

> Note:
>
> - Data in the `frame` column are in the format of `towhee.types.VideoFrame`. These data represent the decoded video frames. For easier understanding, these data are displayed as images in the following figure.
> - This example contains 16 video frames. Not all data are listed in the flat map node step.

![img](https://github.com/towhee-io/data/blob/main/image/docs/window_all_example_2.png?raw=true)



## Notes

- The data type of a merged window is a list. So the function applied by a  window all node should support lists as input.
