import numpy as np
from towhee._types import Image
from towhee.operator import PyOperator


class TritonTestPy(PyOperator):

    def __call__(self, image_path: str):
        return Image(np.random.rand((200, 300, 3), 'BGR'))

    def input_schema(self):
        return [(str, (1,))]

    def output_schema(self):
        return [(Image, (-1, -1, 3))]
