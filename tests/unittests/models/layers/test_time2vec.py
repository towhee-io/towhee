import unittest
from towhee.models.layers.time2vec import Time2Vec
import torch


class TestTime2Vec(unittest.TestCase):
    """
    Test Time2Vec layer
    """
    def test_time2vec(self):
        x = torch.randn(1, 3)
        model1 = Time2Vec(seq_len=3, activation="sin")
        model2 = Time2Vec(seq_len=3, activation="cos")
        self.assertEqual(model1(x).shape, model2(x).shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
