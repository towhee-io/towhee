# Non-local block using embedded gaussian
# Code from
# https://github.com/AlexHex7/Non-local_pytorch/blob/master/Non-Local_pytorch_0.3.1/lib/non_local_embedded_gaussian.py
# modified by Zilliz.

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

class _NonLocalBlockND(nn.Module):
    """
    Args:
    """
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super().__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.w = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.w[1].weight, 0)
            nn.init.constant_(self.w[1].bias, 0)
        else:
            self.w = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.w.weight, 0)
            nn.init.constant_(self.w.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_c = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_c, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        w_y = self.w(y)
        z = w_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    """
    Args:
    """
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__(in_channels,
                        inter_channels=inter_channels,
                        dimension=1, sub_sample=sub_sample,
                        bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    """
    Args:
    """
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__(in_channels,
                        inter_channels=inter_channels,
                        dimension=2, sub_sample=sub_sample,
                        bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    """
    Args:
    """
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__(in_channels,
                        inter_channels=inter_channels,
                        dimension=3, sub_sample=sub_sample,
                        bn_layer=bn_layer)


class NL3DWrapper(nn.Module):
    """
    Args:
    """
    def __init__(self, block, n_segment):
        super().__init__()
        self.block = block
        self.nl = NONLocalBlock3D(block.bn3.num_features)
        self.n_segment = n_segment

    def forward(self, x):
        x = self.block(x)

        nt, c, h, w = x.size()
        x = x.view(nt // self.n_segment, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = self.nl(x)
        x = x.transpose(1, 2).contiguous().view(nt, c, h, w)
        return x


def make_non_local(net, n_segment):
    if isinstance(net, torchvision.models.ResNet):
        net.layer2 = nn.Sequential(
            NL3DWrapper(net.layer2[0], n_segment),
            net.layer2[1],
            NL3DWrapper(net.layer2[2], n_segment),
            net.layer2[3],
        )
        net.layer3 = nn.Sequential(
            NL3DWrapper(net.layer3[0], n_segment),
            net.layer3[1],
            NL3DWrapper(net.layer3[2], n_segment),
            net.layer3[3],
            NL3DWrapper(net.layer3[4], n_segment),
            net.layer3[5],
        )
    else:
        raise NotImplementedError
