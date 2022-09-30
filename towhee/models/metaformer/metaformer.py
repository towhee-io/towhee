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
from functools import partial

from towhee.models.utils.weight_init import trunc_normal_
from towhee.models.poolformer.groupnorm import GroupNorm
from towhee.models.poolformer.layernormchannel import LayerNormChannel
from towhee.models.poolformer.pooling import Pooling
from towhee.models.metaformer.basicblocks import basic_blocks
from towhee.models.poolformer.patchembed import PatchEmbed
from towhee.models.metaformer.addpositionembed import AddPositionEmb
from towhee.models.metaformer.attention import Attention
from towhee.models.metaformer.spatialfc import SpatialFc


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


class MetaFormer(nn.Module):
    """
    MetaFormer
    Args:
        layers (tuple[int]): number of blocks for the 4 stages
        embed_dims (tuple[int]): embedding dimension
        mlp_ratios (tuple[float]): mlp ratios
        token_mixers (tuple[nn.module]): token mixers of different stages
        norm_layer (nn.module): normalization layer
        act_layer (nn.module): activation layer
        num_classes (int): number of classes
        in_patch_size (int): patch embedding size
        in_stride (int): stride of patch embedding
        in_pad (int): padding
        down_patch_size (int): down sample path size
        add_pos_embs (bool): down sample path size
        drop_rate (float): drop rate
        drop_path_rate (float): drop path rate
        use_layer_scale (bool): use layer scale
        layer_scale_init_value (float): layer scale init value
    """
    def __init__(self,
                 layers,
                 embed_dims=None,
                 token_mixers=None,
                 mlp_ratios=None,
                 norm_layer=LayerNormChannel,
                 act_layer=nn.GELU,
                 num_classes=1000,
                 in_patch_size=7,
                 in_stride=4,
                 in_pad=2,
                 downsamples=None,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 add_pos_embs=None,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5,
                 ):

        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=3, embed_dim=embed_dims[0])
        if add_pos_embs is None:
            add_pos_embs = [None] * len(layers)
        if token_mixers is None:
            token_mixers = [nn.Identity] * len(layers)
        # set the main block in network
        network = []
        for i in range(len(layers)):
            if add_pos_embs[i] is not None:
                network.append(add_pos_embs[i](embed_dims[i]))
            stage = basic_blocks(embed_dims[i], i, layers,
                                 token_mixer=token_mixers[i], mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                        )
                    )

        self.network = nn.ModuleList(network)
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 \
            else nn.Identity()

        self.apply(self.cls_init_weights)

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
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        x = self.norm(x)
        # for image classification
        cls_out = self.head(x.mean([-2, -1]))
        return cls_out


model_urls = {
    "metaformer_id_s12": "https://github.com/sail-sg/poolformer/releases/download/v1.0/metaformer_id_s12.pth.tar",
    "metaformer_pppa_s12_224": "https://github.com/sail-sg/poolformer/releases/download/v1.0/metaformer_pppa_s12_224.pth.tar",
    "metaformer_ppaa_s12_224": "https://github.com/sail-sg/poolformer/releases/download/v1.0/metaformer_ppaa_s12_224.pth.tar",
    "metaformer_pppf_s12_224": "https://github.com/sail-sg/poolformer/releases/download/v1.0/metaformer_pppf_s12_224.pth.tar",
    "metaformer_ppff_s12_224": "https://github.com/sail-sg/poolformer/releases/download/v1.0/metaformer_ppff_s12_224.pth.tar",
}


def metaformer_id_s12(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    token_mixers = [nn.Identity] * len(layers)
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_ratios=mlp_ratios,
        norm_layer=GroupNorm,
        downsamples=downsamples,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained:
        url = model_urls['metaformer_id_s12']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


def metaformer_pppa_s12_224(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    add_pos_embs = [None, None, None,
        partial(AddPositionEmb, spatial_shape=[7, 7])]
    token_mixers = [Pooling, Pooling, Pooling, Attention]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        add_pos_embs=add_pos_embs,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['metaformer_pppa_s12_224']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


def metaformer_ppaa_s12_224(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    add_pos_embs = [None, None,
        partial(AddPositionEmb, spatial_shape=[14, 14]), None]
    token_mixers = [Pooling, Pooling, Attention, Attention]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        add_pos_embs=add_pos_embs,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['metaformer_ppaa_s12_224']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


def metaformer_pppf_s12_224(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    token_mixers = [Pooling, Pooling, Pooling,
        partial(SpatialFc, spatial_shape=[7, 7]),
        ]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_ratios=mlp_ratios,
        norm_layer=GroupNorm,
        downsamples=downsamples,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained:
        url = model_urls['metaformer_pppf_s12_224']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


def metaformer_ppff_s12_224(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    token_mixers = [Pooling, Pooling,
        partial(SpatialFc, spatial_shape=[14, 14]),
        partial(SpatialFc, spatial_shape=[7, 7]),
        ]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_ratios=mlp_ratios,
        norm_layer=GroupNorm,
        downsamples=downsamples,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['metaformer_ppff_s12_224']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model
