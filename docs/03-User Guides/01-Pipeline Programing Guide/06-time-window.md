# Time Window

## Introduction

The time window node is used to batch rows that have a time sequence, for example, audio or video frames.

`time_window` is similar to `window`, but the batching rule is applied based on a timestamp column `timestamp_col`. `size` is the time interval of each window, and `step` determines how long a window moves from the previous one. Note that if `step` is less than `size`, the windows will overlap. Refer to [time_window API](/04-API%20Reference/01-Pipeline%20API/07-time-window.md) for more details.

The figure below shows the relationship between `size`, `step`, input rows, and windows. Note that `size` and `step` are both measured by time units. In addition, the video frames may vary in length, so the number of frames in each window can be different.

![img](https://github.com/towhee-io/data/blob/main/image/docs/time_window_intro_1.png?raw=true)

This figure illustrates how `winodw` applies the transformation to the rows:

![img](https://github.com/towhee-io/data/blob/main/image/docs/time_window_intro_2.png?raw=true)

Compared to the window node, the time window node introduces a new parameter `timestamp_col` which specifies a column to include timestamps. The time window node organizes and orders the windows according to the values in the `timestamp_col` column.

The function applied by the window node takes the specified column as input, and the `input_schema` can contain the `timestamp_col` column.



## Example

We use the `time_window(input_schema, output_schema, timestamp_col, size, step, fn, config=None)` interface to create a time window node. `size` is the time interval of each window, and `step` determines how long a window moves from the previous one. Note that when creating a time window node, the upstream input table must contain a column of timestamps (`timestamp_col`). The input of the `fn` function should follow the `input_schema`while  the output of the `fn` function should follow the `output_schema`. 



Now let's take a video frame and feature extraction pipeline to demonstrate how to use a time window node. This pipeline randomly selects one video frame every one second and extracts the feature embeddings of the selected frames. 

```Python
from towhee import ops, pipe
import random

video_frame_embedding = (
    pipe.input('url')
    .flat_map('url', 'frame', ops.video_decode.ffmpeg())
    .map('frame', 'ts', lambda frame: frame.timestamp)
    .time_window('frame', 'frame', 'ts', 1, 1, lambda x: x[random.randint(0, len(x)-1)])
    .map('frame', 'embedding', ops.image_embedding.timm(model_name='resnet50'))
    .output('url', 'frame', 'embedding')
)

data = 'https://raw.githubusercontent.com/towhee-io/examples/0.7/video/reverse_video_search/tmp/Ou1w86qEr58.gif'
res = video_frame_embedding(data)
```

 The DAG of the `video_frame_embedding` pipeline is illustrated below.

![img](https://github.com/towhee-io/data/blob/main/image/docs/time_window_example_1.png?raw=true)

The data processing workflow of the main nodes is as follows.

1. **Flat map:** Uses the [video-decode/ffmpeg](https://towhee.io/video-decode/ffmpeg) operator to decode video URLs (`url`) into a list of video frames (`List(towhee.types.VideoFrame)`), and then flattens this list into multiple rows. Each row contains one video frame.
2. **Map:** Applies the function `lambda frame: frame.timestamp` to obtain the corresponding timestamp (`ts`) of each video frame (`frame`).
3. **Time window:** Specifies the value of both `size` and `step` as `1` to batch video frames (`frame`) whose timestamps (`ts`) are within the same one second  into a time window, and then applies the function `lambda x: x[random.randint(0, len(x)-1)]` to randomly select one video frame (`frame`) from each window. 

	> Note that if there is less than 1 second at the end of a video, the rest of the video frames will still be batched into a window.

4. **Map:** Uses the [image-embedding/timm](https://towhee.io/image-embedding/timm) operator to extract the feature embeddings (`embedding`) of the selected frames (`frame`).



When the pipeline is running, data transformation in each node is illustrated below.

> Note：
>
> - Data in the `frame` column are in the format of `towhee.types.VideoFrame`. These data represent the decoded video frames. For easier understanding, these data are displayed as images in the following figure.
> - This example contains 16 video frames. Not all data are listed in the flat map node step.
> - Since the video frames are randomly selected, the output you get can vary from our example.

![img](https://github.com/towhee-io/data/blob/main/image/docs/time_window_example_2.png?raw=true)



## Notes

- A time window node required a list of timestamps.
- Data in each time window should be lists, so the function applied by a time window node should support list as input.
