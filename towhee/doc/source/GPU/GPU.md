# GPU

In towhee, the operators related to the model basically support GPU. You can pass in the device through the initialization parameters of the operator. If the GPU device is not set, the operator will also try to use the GPU first.

Example:

```python
from towhee import ops, pipe

p = (
    pipe.input('image_file')
    .map('image_file', 'image', ops.image_decode.cv2())
    .map('image', 'vec', ops.image_embedding.timm(model_name='resnet50', device='cuda:0'))
    .output('vec')
)

p('./image.jpg').to_list()
```