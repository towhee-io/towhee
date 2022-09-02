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
import torch.cuda
from torch import nn
import torch.utils.checkpoint as torch_checkpoint

from towhee.models.layers.ffn import FFNBlock
from towhee.models.layers.conv_bn_activation import Conv2dBNActivation
from .blocks import RepMLPBlock
from .configs import get_configs


class RepMLPNetUnit(nn.Module):
    """
    RepMLP Unit (composed of RepMLP block)

    Args:
        channels (`int`):
            Number of input channels & final output channels.
        internal_neurons (`int`):
            Number of channels used to connect conv2d layers inside block.
        h (`int`):
            Input image height.
        w (`int`):
            Input image weight.
        reparam_conv_k (`tuple`):
            Numbers of conv layers.
        globalperceptron_reduce (`int`):
            Number to reduce internal hidden channels.
        ffn_expand (`int`):
            Number to expan channels in FFN block
        num_sharesets (`int`):
            Number of sharesets.
        deploy (`bool`):
            Flag to control deploy parameters like bias.
    """

    def __init__(self, channels, h, w, reparam_conv_k, globalperceptron_reduce, ffn_expand=4,
                 num_sharesets=1, deploy=False):
        super().__init__()
        self.repmlp_block = RepMLPBlock(in_channels=channels, out_channels=channels, h=h, w=w,
                                        reparam_conv_k=reparam_conv_k, globalperceptron_reduce=globalperceptron_reduce,
                                        num_sharesets=num_sharesets, deploy=deploy)
        self.ffn_block = FFNBlock(channels, channels * ffn_expand)
        self.prebn1 = nn.BatchNorm2d(channels)
        self.prebn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = x + self.repmlp_block(self.prebn1(x))
        z = y + self.ffn_block(self.prebn2(y))
        return z


class RepMLPNet(nn.Module):
    """
    RepMLP Net

    Args:
        - in_channels (`int`):
            Number of input channels.
        - num_classes (`int`):
            Number of classes.
        - patch_size (`tuple`):
            Patch size in a tuple (h, w).
        - num_blocks (`tuple`):
            Block numbers used in all stages
        - channels (`int`):
            Numbers of output channels used in all stages.
        - hs (`tuple`):
            Image heights used in all stages.
        - ws (`tuple`):
            Image weights used in all stages.
        - sharesets_nums (`tuple`):
            Shareset_nums used in all stages.
        - reparam_conv_k (`tuple`):
            Numbers of conv layers.
        - globalperceptron_reduce (`int`):
            Number to reduce internal hidden channels.
        - use_checkpoint (`bool`):
            Whether to load checkpoint.
        - deploy (`bool`):
            Flag to control deploy parameters like bias.

    Example:
        >>> from towhee.models.repmlp import RepMLPNet
        >>> import torch
        >>>
        >>> data = torch.rand(1, 3, 1536, 1536)
        >>> model = RepMLPNet()
        >>> outs = model(data)
        >>> print(data.shape)
        torch.Size([1, 1000])
    """
    def __init__(self,
                 in_channels=3, num_class=1000,
                 patch_size=(4, 4),
                 num_blocks=(2, 2, 6, 2), channels=(192, 384, 768, 1536),
                 hs=(64, 32, 16, 8), ws=(64, 32, 16, 8),
                 sharesets_nums=(4, 8, 16, 32),
                 reparam_conv_k=(3,),
                 globalperceptron_reduce=4, use_checkpoint=False,
                 deploy=False):
        super().__init__()
        num_stages = len(num_blocks)
        assert num_stages == len(channels)
        assert num_stages == len(hs)
        assert num_stages == len(ws)
        assert num_stages == len(sharesets_nums)

        self.conv_embedding = Conv2dBNActivation(
            in_planes=in_channels, out_planes=channels[0],
            kernel_size=patch_size, stride=patch_size, padding=0,
            activation_layer=nn.ReLU
        )

        stages = []
        embeds = []
        for stage_idx in range(num_stages):
            stage_blocks = [RepMLPNetUnit(channels=channels[stage_idx], h=hs[stage_idx], w=ws[stage_idx],
                                          reparam_conv_k=reparam_conv_k,
                                          globalperceptron_reduce=globalperceptron_reduce, ffn_expand=4,
                                          num_sharesets=sharesets_nums[stage_idx],
                                          deploy=deploy) for _ in range(num_blocks[stage_idx])]
            stages.append(nn.ModuleList(stage_blocks))
            if stage_idx < num_stages - 1:
                embeds.append(
                    Conv2dBNActivation(
                        in_planes=channels[stage_idx], out_planes=channels[stage_idx + 1],
                        kernel_size=2, stride=2, padding=0,
                        activation_layer=nn.ReLU
                    )
                )

        self.stages = nn.ModuleList(stages)
        self.embeds = nn.ModuleList(embeds)
        self.head_norm = nn.BatchNorm2d(channels[-1])
        self.head = nn.Linear(channels[-1], num_class)

        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        x = self.conv_embedding(x)
        for i, stage in enumerate(self.stages):
            for block in stage:
                if self.use_checkpoint:
                    x = torch_checkpoint.checkpoint(block, x)
                else:
                    x = block(x)
            if i < len(self.stages) - 1:
                embed = self.embeds[i]
                if self.use_checkpoint:
                    x = torch_checkpoint.checkpoint(embed, x)
                else:
                    x = embed(x)
        x = self.head_norm(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

    def locality_injection(self):
        for m in self.modules():
            if hasattr(m, 'local_inject'):
                m.local_inject()


def create_model(
        model_name: str,
        pretrained: bool = False,
        checkpoint_path: str = None,
        device: str = None,
        **kwargs
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_configs(model_name)
    cfg.update(**kwargs)
    model = RepMLPNet(**cfg).to(device)

    if pretrained:
        assert checkpoint_path, 'Checkpoint path is mandatory for pretrained model.'
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    model.eval()
    return model
