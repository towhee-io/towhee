# Original pytorch implementation by:
# 'MaxViT: Multi-Axis Vision Transformer'
#       - https://arxiv.org/pdf/2204.01697.pdf
# Original code by / Copyright 2021, Christoph Reich.
# Modifications & additions by / Copyright 2022 Zilliz. All rights reserved.
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
from typing import Tuple, Optional
import torch
from torch import nn
from towhee.models.max_vit.max_vit_block import MaxViTStage
from towhee.models.max_vit.configs import get_configs


class MaxViT(nn.Module):
    """
    Implementation of the MaxViT proposed in:
    https://arxiv.org/pdf/2204.01697.pdf

    Args:
        in_channels (`int`):
            Number of input channels to the convolutional stem. Default 3
        depths (`Tuple[int, ...]`):
            Depth of each network stage. Default (2, 2, 5, 2)
        channels (`Tuple[int, ...]`):
            Number of channels in each network stage. Default (64, 128, 256, 512)
        num_classes (`int`):
            Number of classes to be predicted. Default 1000
        embed_dim (`int`):
            Embedding dimension of the convolutional stem. Default 64
        num_heads (`int`):
            Number of attention heads. Default 32
        grid_window_size (`Tuple[int, int]`):
            Grid/Window size to be utilized. Default (7, 7)
        attn_drop (`float`):
            Dropout ratio of attention weight. Default: 0.0
        drop (`float`):
            Dropout ratio of output. Default: 0.0
        drop_path (`float`):
            Dropout ratio of path. Default: 0.0
        mlp_ratio (`float`):
            Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (`nn.Module`):
            Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (`nn.Module`):
            Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        norm_layer_transformer (`nn.Module`):
            Normalization layer in Transformer. Default: nn.LayerNorm
        global_pool (`str`):
            Global polling type to be utilized. Default "avg"
    """

    def __init__(
            self,
            in_channels: int = 3,
            depths: Tuple[int, ...] = (2, 2, 5, 2),
            channels: Tuple[int, ...] = (64, 128, 256, 512),
            num_classes: int = 1000,
            embed_dim: int = 64,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop : float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer=nn.GELU,
            norm_layer=nn.BatchNorm2d,
            norm_layer_transformer=nn.LayerNorm,
            global_pool: str = "avg"
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super().__init__()
        # Check parameters
        assert len(depths) == len(channels), "For each stage a channel dimension must be given."
        assert global_pool in ["avg", "max"], f"Only avg and max is supported but {global_pool} is given"
        # Save parameters
        self.num_classes: int = num_classes
        # Init convolutional stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=(3, 3), stride=(2, 2),
                      padding=(1, 1)),
            act_layer(),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            act_layer(),
        )
        # Init blocks
        drop_path = torch.linspace(0.0, drop_path, sum(depths)).tolist()
        stages = []
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            stages.append(
                MaxViTStage(
                    depth=depth,
                    in_channels=embed_dim if index == 0 else channels[index - 1],
                    out_channels=channel,
                    num_heads=num_heads,
                    grid_window_size=grid_window_size,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    norm_layer_transformer=norm_layer_transformer
                )
            )
        self.stages = nn.Sequential(*stages)
        self.global_pool: str = global_pool
        self.head = nn.Linear(channels[-1], num_classes)

    # @torch.jit.ignore
    # def no_weight_decay(self) -> Set[str]:
    #     """ Gets the names of parameters to not apply weight decay to.
    #     Returns:
    #         nwd (Set[str]): Set of parameter names to not apply weight decay to.
    #     """
    #     nwd = set()
    #     for n, _ in self.named_parameters():
    #         if "relative_position_bias_table" in n:
    #             nwd.add(n)
    #     return nwd

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """
        Method results the classification head
        Args:
            num_classes (`int`):
                Number of classes to be predicted.
            global_pool (`str`):
                If not global pooling is updated.
        """
        self.num_classes: int = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feature extraction.
        Args:
            data (`torch.Tensor`):
                Input images of the shape [B, C, H, W].
        Returns:
            output (`torch.Tensor`):
                Image features of the backbone.
        """
        output = data
        for stage in self.stages:
            output = stage(output)
        return output

    def forward_head(self, data: torch.Tensor, pre_logits: bool = False):
        """
        Forward pass of classification head.
        Args:
            data (`torch.Tensor`):
                Input features.
            pre_logits (`bool`):
                If true pre-logits are returned.
        Returns:
            output (`torch.Tensor`):
                Classification output of the shape [B, num_classes].
        """
        if self.global_pool == "avg":
            data = data.mean(dim=(2, 3))
        elif self.global_pool == "max":
            data = torch.amax(data, dim=(2, 3))
        return data if pre_logits else self.head(data)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """ Forward pass
        Args:
            data (`torch.Tensor`):
                Input images of the shape [B, C, H, W].
        Returns:
            output (`torch.Tensor`):
                Classification output of the shape [B, num_classes].
        """
        output = self.forward_features(self.stem(data))
        output = self.forward_head(output)
        return output


def create_model(
        model_name: str = None,
        pretrained: bool = False,
        weights_path: str = None,
        device: str = None,
        **kwargs
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if pretrained and weights_path is None:
        raise AssertionError("if pretrained is true, weights_path needs to be specified")
    if model_name is None:
        if pretrained:
            raise AssertionError("Fail to load pretrained model: no model name is specified.")
        model = MaxViT(**kwargs)
    else:
        configs = get_configs(model_name)
        configs.update(kwargs)
        model = MaxViT(**configs)
        if pretrained:
            state_dic = torch.load(weights_path, map_location=device)["model_state"]
            model.load_state_dict(state_dic)

    model.eval()
    model.to(device)
    return model


# if __name__ == '__main__':
#     data = torch.rand(1, 3, 224, 224)
#     model = create_model(model_name='max_vit_tiny', drop_path=0.2)
