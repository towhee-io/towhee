# Pytorch implementation of [RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality]
# (https://arxiv.org/abs/2112.11081)
#
# Inspired by https://github.com/DingXiaoH/RepMLP
#
# Additions & modifications by Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from towhee.models.layers.conv_bn_activation import Conv2dBNActivation
from towhee.models.utils.fuse_bn import fuse_bn


class GlobalPerceptron(nn.Module):
    """
    Global Perception Block

    Args:
        - input_channels (`int`):
            Number of input channels & final output channels.
        - internal_neurons (`int`):
            Number of channels used to connect conv2d layers inside block.

    Example:
        >>> import torch
        >>> from towhee.models.repmlp import GlobalPerceptron
        >>>
        >>> data = torch.rand(3, 1, 1)
        >>> layer = GlobalPerceptron(input_channels=3, internal_neurons=4)
        >>> out = layer(data)
        >>> print(out.shape)
        torch.Size([1, 3, 1, 1])
    """
    def __init__(self, input_channels, internal_neurons):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = nn.functional.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.fc1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return x


class RepMLPBlock(nn.Module):
    """
    RepMLP Block.

    Args:
        input_channels (`int`):
            Number of input channels & final output channels.
        internal_neurons (`int`):
            Number of channels used to connect conv2d layers inside block.
        h (`int`):
            Input image height.
        w (`int`):
            Input image weight.
        reparam_conv_k (`tuple or list`):
            Numbers of conv layers.
        globalperceptron_reduce (`int`):
            Number to reduce internal hidden channels.
        num_sharesets (`int`):
            Number of sharesets.
        deploy (`bool`):
            Flag to control deploy parameters like bias.

    Example:
        >>> from towhee.models.repmlp import RepMLPBlock
        >>> import torch
        >>>
        >>> data = torch.rand(1, 4, 6, 6)
        >>> model = RepMLPBlock(in_channels=4, out_channels=4, h=3, w=3)
        >>> outs = model(data)
        >>> print(outs.shape)
        torch.Size([1, 4, 6, 6])
    """
    def __init__(self, in_channels, out_channels,
                 h, w,
                 reparam_conv_k=None,
                 globalperceptron_reduce=4,
                 num_sharesets=1,
                 deploy=False):
        super().__init__()

        self.in_c = in_channels
        self.out_c = out_channels
        self.share_s = num_sharesets
        self.h, self.w = h, w
        self.deploy = deploy

        assert in_channels == out_channels
        self.gp = GlobalPerceptron(input_channels=in_channels, internal_neurons=in_channels // globalperceptron_reduce)

        self.fc3 = nn.Conv2d(
            self.h * self.w * num_sharesets,
            self.h * self.w * num_sharesets,
            1, 1, 0,
            bias=deploy, groups=num_sharesets)
        if deploy:
            self.fc3_bn = nn.Identity()
        else:
            self.fc3_bn = nn.BatchNorm2d(num_sharesets)

        self.reparam_conv_k = reparam_conv_k
        if not deploy and reparam_conv_k is not None:
            for k in reparam_conv_k:
                conv_branch = Conv2dBNActivation(
                    num_sharesets, num_sharesets, kernel_size=k, stride=1, padding=k//2, groups=num_sharesets,
                    norm_layer=nn.BatchNorm2d, eps=1e-5
                )
                self.__setattr__(f'repconv{k}', conv_branch)

    def partition(self, x, h_parts, w_parts):
        x = x.reshape(-1, self.in_c, h_parts, self.h, w_parts, self.w)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x

    def partition_affine(self, x, h_parts, w_parts):
        fc_inputs = x.reshape(-1, self.share_s * self.h * self.w, 1, 1)
        out = self.fc3(fc_inputs)
        out = out.reshape(-1, self.share_s, self.h, self.w)
        out = self.fc3_bn(out)
        out = out.reshape(-1, h_parts, w_parts, self.share_s, self.h, self.w)
        return out

    def forward(self, inputs):
        # Global Perceptron
        global_vec = self.gp(inputs)

        origin_shape = inputs.size()
        h_parts = origin_shape[2] // self.h
        w_parts = origin_shape[3] // self.w

        partitions = self.partition(inputs, h_parts, w_parts)

        #   Channel Perceptron
        fc3_out = self.partition_affine(partitions, h_parts, w_parts)

        #   Local Perceptron
        if self.reparam_conv_k is not None and not self.deploy:
            conv_inputs = partitions.reshape(-1, self.share_s, self.h, self.w)
            conv_out = 0
            for k in self.reparam_conv_k:
                conv_branch = self.__getattr__(f'repconv{k}')
                conv_out += conv_branch(conv_inputs)
            conv_out = conv_out.reshape(-1, h_parts, w_parts, self.share_s, self.h, self.w)
            fc3_out += conv_out

        fc3_out = fc3_out.permute(0, 3, 1, 4, 2, 5)  # N, out_c, h_parts, out_h, w_parts, out_w
        out = fc3_out.reshape(*origin_shape)
        out = out * global_vec
        return out

    def get_equivalent_fc3(self):
        fc_weight, fc_bias = fuse_bn(self.fc3, self.fc3_bn)
        if self.reparam_conv_k is not None:
            largest_k = max(self.reparam_conv_k)
            largest_branch = self.__getattr__(f'repconv{largest_k}')
            total_kernel, total_bias = fuse_bn(largest_branch.conv2d, largest_branch.norm)
            for k in self.reparam_conv_k:
                if k != largest_k:
                    k_branch = self.__getattr__(f'repconv{k}')
                    kernel, bias = fuse_bn(k_branch.conv2d, k_branch.norm)
                    total_kernel += nn.functional.pad(kernel, [(largest_k - k) // 2] * 4)
                    total_bias += bias
            rep_weight, rep_bias = self._convert_conv_to_fc(total_kernel, total_bias)
            final_fc3_weight = rep_weight.reshape_as(fc_weight) + fc_weight
            final_fc3_bias = rep_bias + fc_bias
        else:
            final_fc3_weight = fc_weight
            final_fc3_bias = fc_bias
        return final_fc3_weight, final_fc3_bias

    def local_inject(self):
        self.deploy = True
        # Locality Injection
        fc3_weight, fc3_bias = self.get_equivalent_fc3()
        # Remove Local Perceptron
        # if self.reparam_conv_k is not None:
        #     for k in self.reparam_conv_k:
        #         self.__delattr__(f'repconv{k}')
        self.__delattr__('fc3')
        self.__delattr__('fc3_bn')
        self.fc3 = nn.Conv2d(
            self.share_s * self.h * self.w,
            self.share_s * self.h * self.w,
            1, 1, 0,
            bias=True, groups=self.share_s)
        self.fc3_bn = nn.Identity()
        self.fc3.weight.data = fc3_weight
        self.fc3.bias.data = fc3_bias

    def _convert_conv_to_fc(self, conv_kernel, conv_bias):
        inputs = torch.eye(self.h * self.w).repeat(1, self.share_s).reshape(
            self.h * self.w, self.share_s, self.h, self.w).to(conv_kernel.device)
        fc_k = nn.functional.conv2d(
            inputs, conv_kernel, padding=(conv_kernel.size(2)//2,conv_kernel.size(3)//2), groups=self.share_s)
        fc_k = fc_k.reshape(self.h * self.w, self.share_s * self.h * self.w).t()
        fc_bias = conv_bias.repeat_interleave(self.h * self.w)
        return fc_k, fc_bias

