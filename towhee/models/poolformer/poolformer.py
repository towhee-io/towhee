# Built on top of the original implementation at https://github.com/sail-sg/poolformer/blob/main/models/poolformer.py
#
# Modifications by Copyright 2022 Zilliz. All rights reserved.
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

from torch import nn
import torch
import os
import copy

from towhee.models.poolformer.patchembed import PatchEmbed
from towhee.models.utils.weight_init import trunc_normal_

from towhee.models.poolformer.basic_blocks import basic_blocks
from towhee.models.poolformer.groupnorm import GroupNorm


class PoolFormer(nn.Module):
    """
    PoolFormer
    Args:
        layers (tuple[int]): number of blocks for the 4 stages
        embed_dims (tuple[int]): embedding dimension
        mlp_ratios (tuple[float]): mlp ratios
        pool_size (int): pooling size
        norm_layer (nn.module): normalization layer
        act_layer (nn.module): activation layer
        num_classes (int): number of classes
        in_patch_size (int): patch embedding size
        in_stride (int): stride of patch embedding
        in_pad (int): padding
        down_patch_size (int): down sample path size
        down_pad (int): down sample padding
        drop_rate (float): drop rate
        drop_path_rate (float): drop path rate
        use_layer_scale (bool): use layer scale
        layer_scale_init_value (float): layer scale init value
        fork_feat (bool): layer scale init value
        init_cfg (dict): config for init
        pretrained (bool): pretrained
    """

    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None, downsamples=None,
                 pool_size=3,
                 norm_layer=GroupNorm, act_layer=nn.GELU,
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=3, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out


model_urls = {
    "poolformer_s12": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar",
    "poolformer_s24": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s24.pth.tar",
    "poolformer_s36": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s36.pth.tar",
    "poolformer_m36": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m36.pth.tar",
    "poolformer_m48": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m48.pth.tar",
}


def poolformer_s12(pretrained=False, **kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    """
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    if pretrained:
        url = model_urls['poolformer_s12']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'poolformer_s': _cfg(crop_pct=0.9),
    'poolformer_m': _cfg(crop_pct=0.95),
}


def poolformer_s24(pretrained=False, **kwargs):
    """
    PoolFormer-S24 model, Params: 21M
    """
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    if pretrained:
        url = model_urls['poolformer_s24']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


def poolformer_s36(pretrained=False, **kwargs):
    """
    PoolFormer-S36 model, Params: 31M
    """
    layers = [6, 6, 18, 6]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    if pretrained:
        url = model_urls['poolformer_s36']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


def poolformer_m36(pretrained=False, **kwargs):
    """
    PoolFormer-M36 model, Params: 56M
    """
    layers = [6, 6, 18, 6]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_m']
    if pretrained:
        url = model_urls['poolformer_m36']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model
