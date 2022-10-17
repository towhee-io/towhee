# Pytorch implementation of RepLKNet model in the paper:
#       "Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs"
#       (https://arxiv.org/abs/2203.06717)
#
# Inspired by the original code from https://github.com/DingXiaoH/RepLKNet-pytorch
#
# All additions & modifications by Copyright 2021 Zilliz. All rights reserved.
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

import os

import torch
from torch import nn

from towhee.models.utils.download import download_from_url
from towhee.models.replknet import RepLKNetStage, fuse_bn, conv_bn_relu, get_configs


class RepLKNet(nn.Module):
    """
    RepLKNet model

    Args:
        large_kernel_sizes `tuple or list`: large kernel sizes
        layers (`tuple or list`): number of blocks at different stages
        channels (`tuple or list`): dimensions used at different stages
        drop_rate (`float`): drop rate used for stochastic depth
        small_kernel (`int`): the small kernel size
        dw_ratio (`int`): times of dim over input dim in depthwise conv
        ffn_ratio (`int`): times of internal dim over input dim in conv FFN
        in_channels (`int`): input dimension
        num_classes (`int`): number of classes for linear projection at the end
        out_indices (`tuple or list`): layer indices to return outputs
        small_kernel_merged (`bool`): flag to merge small kernels in ReparamLargeKernelConv
        norm_intermediate_features (`bool`): flag to return normalized features for downstream tasks
        deep_fuse (`bool`): flag to manually fuse BN
    """

    def __init__(self, large_kernel_sizes, layers, channels, drop_rate, small_kernel,
                 dw_ratio=1, ffn_ratio=4, in_channels=3, num_classes=1000, out_indices=None,
                 small_kernel_merged=False, norm_intermediate_features=False, deep_fuse=True
                 ):
        super().__init__()
        self.large_kernel_sizes = large_kernel_sizes
        self.layers = layers
        self.channels = channels
        self.num_classes = num_classes
        self.small_kernel = small_kernel
        self.dw_ratio = dw_ratio
        self.deep_fuse = deep_fuse

        if num_classes is None and out_indices is None:
            raise ValueError('must specify one of num_classes (for pretraining) and out_indices (for downstream tasks)')
        elif num_classes is not None and out_indices is not None:
            raise ValueError('cannot specify both num_classes (for pretraining) and out_indices (for downstream tasks)')
        elif num_classes is not None and norm_intermediate_features:
            raise ValueError('for pretraining, no need to normalize the intermediate feature maps')
        self.out_indices = out_indices

        base_width = channels[0]
        self.norm_intermediate_features = norm_intermediate_features
        self.num_stages = len(layers)
        self.stem = nn.ModuleList([
            conv_bn_relu(
                in_channels=in_channels, out_channels=base_width,
                kernel_size=3, stride=2, padding=1, groups=1
            ),
            conv_bn_relu(
                in_channels=base_width, out_channels=base_width,
                kernel_size=3, stride=1, padding=1, groups=base_width
            ),
            conv_bn_relu(
                in_channels=base_width, out_channels=base_width,
                kernel_size=1, stride=2, padding=0, groups=1
            ),
            conv_bn_relu(
                in_channels=base_width, out_channels=base_width,
                kernel_size=3, stride=2, padding=1, groups=base_width
            )])

        # Stochastic depth. Set block-wise drop-path rate.
        # The higher level blocks are more likely to be dropped. This implementation follows Swin.
        dpr = [x.item() for x in torch.linspace(0, drop_rate, sum(layers))]
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            layer = RepLKNetStage(channels=channels[stage_idx], num_blocks=layers[stage_idx],
                                  stage_lk_size=large_kernel_sizes[stage_idx],
                                  drop_rate=dpr[sum(layers[:stage_idx]):sum(layers[:stage_idx + 1])],
                                  small_kernel=small_kernel, dw_ratio=dw_ratio, ffn_ratio=ffn_ratio,
                                  small_kernel_merged=small_kernel_merged,
                                  norm_intermediate_features=norm_intermediate_features)
            self.stages.append(layer)
            if stage_idx < len(layers) - 1:
                transition = nn.Sequential(
                    conv_bn_relu(
                        in_channels=channels[stage_idx], out_channels=channels[stage_idx + 1],
                        kernel_size=1, stride=1, padding=0, groups=1
                    ),
                    conv_bn_relu(
                        in_channels=channels[stage_idx + 1], out_channels=channels[stage_idx + 1],
                        kernel_size=3, stride=2, padding=1, groups=channels[stage_idx + 1]
                    ))
                self.transitions.append(transition)

        if num_classes is not None:
            self.norm = nn.BatchNorm2d(channels[-1])
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(channels[-1], num_classes)

    def forward_features(self, x):
        # Manually fuse BN for inference
        if self.deep_fuse:
            self.deep_fuse_bn()

        x = self.stem[0](x)

        for stem_layer in self.stem[1:]:
            x = stem_layer(x)
        if self.out_indices is None:
            # Just need the final output
            for stage_idx in range(self.num_stages):
                x = self.stages[stage_idx](x)
                if stage_idx < self.num_stages - 1:
                    x = self.transitions[stage_idx](x)
            return x
        else:
            # Need the intermediate feature maps
            outs = []
            for stage_idx in range(self.num_stages):
                x = self.stages[stage_idx](x)
                if stage_idx in self.out_indices:
                    outs.append(self.stages[stage_idx].norm(
                        x))  # For RepLKNet-XL normalize the features before feeding them into the heads
                if stage_idx < self.num_stages - 1:
                    x = self.transitions[stage_idx](x)
            return outs

    def forward(self, x):
        x = self.forward_features(x)
        if self.out_indices:
            return x
        else:
            x = self.norm(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.head(x)
            return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

    def deep_fuse_bn(self):
        for m in self.modules():
            if not isinstance(m, nn.Sequential):
                continue
            if not len(m) in [2, 3]:  # Only handle conv-BN or conv-BN-relu
                continue
            # If you use a custom Conv2d impl, assume it also has 'kernel_size' and 'weight'
            if hasattr(m[0], 'kernel_size') and hasattr(m[0], 'weight') and isinstance(m[1], nn.BatchNorm2d):
                conv = m[0]
                bn = m[1]
                fused_kernel, fused_bias = fuse_bn(conv, bn)
                fused_conv = nn.Conv2d(
                    in_channels=conv.in_channels, out_channels=conv.out_channels,
                    kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=conv.groups, bias=True)
                fused_conv.weight.data = fused_kernel
                fused_conv.bias.data = fused_bias
                m[0] = fused_conv
                m[1] = nn.Identity()


def create_model(
        model_name: str = None,
        pretrained: bool = False,
        checkpoint_path: str = None,
        device: str = None,
        **kwargs):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configs = get_configs(model_name)

    url = None
    if 'url' in configs:
        url = configs['url']
        del configs['url']
    configs.update(**kwargs)

    model = RepLKNet(**configs).to(device)
    if pretrained:
        if checkpoint_path:
            local_path = checkpoint_path
        elif url:
            cache_dir = os.path.expanduser('~/.cache/towhee')
            local_path = download_from_url(url=url, root=cache_dir)
        else:
            raise AttributeError('No url or checkpoint_path is provided for pretrained model.')
        state_dict = torch.load(local_path, map_location=device)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)
    model.eval()
    return model


# if __name__ == '__main__':
#     import torch
#     x = torch.ones(1, 3, 384, 384)
#     model = create_model(model_name='replknet_31b_1k',
#                          pretrained=True, checkpoint_path='path/to/checkpoint.pth')
#     outs = model(x)
#     print(outs.shape)
