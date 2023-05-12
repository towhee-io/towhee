# Triton in Towhee

[Triton](https://github.com/triton-inference-server) is an inference serving software that streamlines AI inferencing, and Towhee uses Triton to provide model inference and acceleration. 

For example, we tested the performance of two CLIP pipelines on the same machine (64 cores, GeForce RTX 3080), one based on HuggingFace and the other using Towhee&Triton, Towhee is 5x faster than Huggingface.

![640](./qps.png)

## Prerequisites

- [Towhee](https://github.com/towhee-io/towhee)
- [Docker](https://docs.docker.com/engine/install/)

## Example

There is an example of using towhee to start a triton server with a text image search pipeline, you can also refer to this to start your own pipeline.

- **Create Pipeline and Build Image**

When creating a pipeline, we can specify `config` with [AutoConfig](../../../runtime/auto_config.py) to set the configuration. It will work when starting the Triton Model, and the following example shows how to create a pipeline in Triton with `config = AutoConfig.TritonGPUConfig()`.

```Python
from towhee import pipe, ops, AutoConfig

p = (
    pipe.input('url')
    .map('url', 'image', ops.image_decode.cv2_rgb())
    .map('image', 'vec', ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='image'), config=AutoConfig.TritonGPUConfig())
    .output('vec')
)

towhee.build_docker_image(
    dc_pipeline=p,
    image_name='clip:v1',
    cuda_version='11.7', # '117dev' for developer
    format_priority=['onnx'],
    parallelism=4,
    inference_server='triton'
)
```

Then we can run `docker images` command and will list the built **clip:v1** image.

- **Start Triton Server**

Run docker image with `tritonserver` command to start triton server.

```Bash
$ docker run -td --gpus=all --shm-size=1g \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    clip:v1 \
    tritonserver --model-repository=/workspace/models
```

After starting the server, we can run `docker logs <container id>` to view the logs. When you see the following logs, it means that the server started successfully:

```bash
Started GRPCInferenceService at 0.0.0.0:8001
Started HTTPService at 0.0.0.0:8000
Started Metrics Service at 0.0.0.0:8002
```

- **Remote Serving**

The we can use `triton_client` to request the result with the url.

> The url format is your-ip-address:your-grpc-port, you need modify the ip according you env.

```Python
from towhee import triton_client

# run with triton client
client = triton_client.Client(url='localhost:8000')

# run data
data = 'https://github.com/towhee-io/towhee/raw/main/towhee_logo.png'
res = client(data)
```

## Advanced

### Pipeline Configuration

When building the image, we need to specify the following parameters:

- `dc_pipeline`: towhee pipeline
- `image_name`: the name of the image
- `cuda_verion`: the version of CUDA
- `format_priority`: the priority list of the model, defaults to ['onnx']
- `inference_server`: the inference server, defaults to 'triton'

### Docker Configuration

You can set the [Docker Command Options](https://docs.docker.com/engine/reference/commandline/run/) when start triton server, such as set gpus and she-size.

## Q&A

1. **Why the docker image is very large?**

   The base image (nvidia/tritonserver) itself is large, and there are many packages related to the package model (PyTorch, Onnxruntime, etc.) that need to be installed.

2. **How to debug my pipeline in triton?**

   Once you have started triton server, you can run into the container to modify the code with `docker exec -ti <container_id> bash` command, and then manually start the triton service in container with `tritonserver --model-repository=/workspace/models --grpc-port=<your-port>`.

   