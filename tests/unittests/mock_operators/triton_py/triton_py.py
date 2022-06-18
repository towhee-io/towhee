import numpy as np
from towhee._types import Image
from towhee.operator import PyOperator, SharedType
from towhee import register


@register(
    input_schema=[(str, (-1,))],
    output_schema=[(Image, (-1, -1, 3))]
)
class TritonPy(PyOperator):
    def __init__(self):
        super().__init__()

    def __call__(self, image_path: str):
        return Image(np.random.rand((300, 300, 3), 'BGR'))
