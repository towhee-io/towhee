import torch
from torch import nn
import unittest

from towhee.models.layers.conv2_conv1 import Conv2and1D
from towhee.models.layers.create_conv2_conv1 import create_conv_2p_1d


class Conv2and1Test(unittest.TestCase):
    def test_conv2_conv1(self):
        input_dim = 2
        output_dim = 4
        model = Conv2and1D(
            conv_t=nn.Conv3d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
            norm=nn.BatchNorm3d(output_dim),
            activation=nn.ReLU(),
            conv_xy=nn.Conv3d(
                in_channels=output_dim,
                out_channels=output_dim,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                bias=False,
            ),
        )
        model_gt = create_conv_2p_1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
            bias=False,
            norm=nn.BatchNorm3d,
            norm_eps=1e-5,
            norm_momentum=0.1,
            activation=nn.ReLU,
        )
        model.load_state_dict(
            model_gt.state_dict(), strict=True
        )
        tensor = torch.rand((1, input_dim, 3, 7, 7))
        with torch.no_grad():
            output_tensor = model(tensor)
            output_tensor_gt = model_gt(tensor)
        self.assertTrue(output_tensor.shape == output_tensor_gt.shape)
