import unittest

import torch
from torch import nn

from towhee.serving.torch_model_handler import TorchModelHandler


class TestTorchModelWorker(unittest.TestCase):
    """
    TestTorchModelWorker
    """

    def test_cpu_model(self):
        model = nn.Linear(10, 5)
        data = [torch.randn(10) for _ in range(5)]
        modelworker = TorchModelHandler(model)
        output = modelworker(data)
        self.assertEqual(len(output), 5)

    def test_gpu_model(self):
        if not torch.cuda.is_available():
            return
        model = nn.Linear(10, 5)
        data = [torch.randn(10) for _ in range(5)]
        modelworker = TorchModelHandler(model, 0)
        output = modelworker(data)
        self.assertEqual(len(output), 5)


if __name__ == '__main__':
    unittest.main()
