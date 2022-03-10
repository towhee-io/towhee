import unittest
from towhee.models.layers.time2vec import Time2Vec
import torch


class TestTime2Vec(unittest.TestCase):
    """
    Test Time2Vec layer
    """
    def test_time2vec(self):
        x = torch.randn(3, 64)
        model1 = Time2Vec(seq_len=64, activation="sin")
        model2 = Time2Vec(seq_len=64, activation="cos")
        self.assertEqual(model1(x).shape, model2(x).shape, (3, 64))
        