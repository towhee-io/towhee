# Window

## Introduction

The window node batches the input rows into multiple rows based on the specified window size (`size`) and steps (`step`). The `size` determines the window length, and `step` determines how long a window moves from the previous one. Note that if `step` is less than `size`, the windows will overlap. The window node applies a function `func` to each of the windowed data, and returns the results - one row of results for each of the windows. Refer to [window API](/04-API%20Reference/01-Pipeline%20API/06-window.md) for more details.

This figure shows the relationship between `size`, `step`, input rows, and windows：

![img](https://github.com/towhee-io/data/blob/main/image/docs/window_intro_1.png?raw=true)

The figure below illustrates how `winodw` applies the transformation to the rows:

![img](https://github.com/towhee-io/data/blob/main/image/docs/window_intro_2.png?raw=true)

Since rows in each `window` will be organized as a list, the input of a window node function should be a list.



## Example

We use the `window(input_schema, output_schema, size, step, fn, config=None)` interface to create a window node.



Now let's take a video frame and image feature extraction pipeline to demonstrate how to use a window node. This pipeline randomly selects one video frame out of ten frames and extracts the feature embeddings of the selected frames. 

```Python
from towhee import ops, pipe
import random

video_frame_embedding = (
    pipe.input('url')
        .flat_map('url', 'frame', ops.video_decode.ffmpeg())
        .window('frame', 'frame', 10, 10, lambda x: x[random.randint(0, len(x)-1)])
        .map('frame', 'embedding', ops.image_embedding.timm(model_name='resnet50'))
        .output('url', 'frame', 'embedding')
)

data = 'https://raw.githubusercontent.com/towhee-io/examples/0.7/video/reverse_video_search/tmp/Ou1w86qEr58.gif'
res = video_frame_embedding(data)
```

 The DAG of the `video_frame_embedding` pipeline is illustrated below. 

![img](https://github.com/towhee-io/data/blob/main/image/docs/window_example_1.png?raw=true)

The data processing workflow of the main nodes is as follows.

1. **Flat map:** Uses the [video-decode/ffmpeg](https://towhee.io/video-decode/ffmpeg) operator to decode video URLs (`url`) into a list of video frames (`List(towhee.types.VideoFrame)`), and then flattens this list into multiple rows. Each row contains one video frame.
2. **Window:** Specifies the value of both `size` and `step` as `10` to batch every 10 video frames (`frame`) into a window, and then applies the function `lambda x: x[random.randint(0, len(x)-1)]` to randomly select one video frame (`frame`) from each window. 
	> Note that if there are less than ten frames at the end of a video, the rest of the video frames will still be batched into a window.

3. **Map:** Uses the [image-embedding/timm](https://towhee.io/image-embedding/timm) operator to extract the feature embeddings (`embedding`) of the selected frames (`frame`).



When the pipeline is running, data transformation in each node is illustrated below.

> Note:
>
> - Data in the `frame` column are in the format of `towhee.types.VideoFrame`. These data represent the decoded video frames. For easier understanding, these data are displayed as images in the following figure.
> - This example contains 16 video frames. Not all data are listed in the flat map node step.
> - Since the video frames are randomly selected, the output you get can vary from our example.

![img](https://github.com/towhee-io/data/blob/main/image/docs/window_example_2.png?raw=true)



## Notes

- The input of a window node function should be a list.
