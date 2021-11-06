import torch
from torch import nn
import unittest

from towhee.models.layers.conv3d_sum import SumConv3D


class Conv3DTest(unittest.TestCase):
    def test_conv3d(self):
        input_dim = 2
        output_dim = 4
        model = SumConv3D(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=((1, 1, 1), (3, 3, 3), (1, 3, 3)),
            stride=((1, 1, 1), (1, 1, 1), None),
            padding=((0, 0, 0), (1, 1, 1), (0, 1, 1)),
            dilation=((2, 2, 2), (1, 1, 1), None),
            groups=(1, 2, None),
            bias=(True, False, None),
        )
        model_gt_list = [
            nn.Conv3d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                dilation=(2, 2, 2),
                groups=1,
                bias=True,
            ),
            nn.Conv3d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                dilation=(1, 1, 1),
                groups=2,
                bias=False,
            ),
            nn.Conv3d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            ),
        ]
        model.convs[0].load_state_dict(
            model_gt_list[0].state_dict(), strict=True
        )
        model.convs[1].load_state_dict(
            model_gt_list[1].state_dict(), strict=True
        )
        model.convs[2].load_state_dict(
            model_gt_list[2].state_dict(), strict=True
        )
        tensor = torch.rand((1, input_dim, 3, 7, 7))
        if tensor.shape[1] == input_dim:
            output_tensor = model(tensor)
            output_gt = []
            for ind in range(3):
                output_gt.append(model_gt_list[ind](tensor))
            output_tensor_gt = torch.stack(output_gt, dim=0).sum(
                dim=0, keepdim=False
            )
        self.assertTrue(output_tensor.shape == output_tensor_gt.shape)
