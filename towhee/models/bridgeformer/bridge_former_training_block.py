# Original pytorch implementation by:
# 'Frozen in Time: A Joint Image and Video Encoder for End-to-End Retrieval'
#       - https://arxiv.org/abs/2104.00650
# Original code by / Copyright 2021, Max Bain.
# Modifications & additions by / Copyright 2021 Zilliz. All rights reserved.
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

from collections import OrderedDict
from functools import partial
import torch
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from torch import einsum, nn
from towhee.models.layers.mlp import Mlp
from towhee.models.layers.patch_embed2d import PatchEmbed2D
from towhee.models.utils.init_vit_weights import init_vit_weights
from towhee.models.frozen_in_time.frozen_video_transformer import attn


def attn_mask(q, k, v, mask):
    sim = einsum('b i d, b j d -> b i j', q, k)
    mask = (1.0 - mask) * -10000.0
    mask = repeat(mask, 'b d -> b r d', r=q.shape[1])
    sim = sim + mask
    attn2 = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn2, v)
    return out


class VarAttention(nn.Module):
    """
    Modified multi-head Attention.

    Args:
        dim (`int`):
            Dimension of features.
        num_heads (`int`):
            Number of heads.
        qkv_bias (`bool`):
            Flag to control if add bias to qkv layer.
        qk_scale (`float`):
            Number to scale qk.
        attn_drop (`float`):
            Drop rate of attention layer.
        proj_drop (`float`):
            Drop rate of projection layer.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 initialize='random'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if initialize == 'zeros':
            self.qkv.weight.data.fill_(0)
            self.qkv.bias.data.fill_(0)
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            self.proj.weight.data.fill_(1)
            self.proj.bias.data.fill_(0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask, whether, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        # project x to q, k, v values
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q = q * self.scale
        mask = repeat(mask, 'b d -> (b r) d', r=self.num_heads)
        n_f = int(einops_dims['f'])
        # splice out CLS token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))
        # let CLS token attend to key / values of all patches across time and space
        if whether is not True:
            cls_out = attn(cls_q, k, v)
        else:
            cls_mask = mask[:, 0:1]
            mask_ = mask[:, 1:]
            mask_ = mask_.repeat(1, n_f)
            mask_tile = torch.cat((cls_mask, mask_), dim=1)
            cls_out = attn_mask(cls_q, k, v, mask_tile)
        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        if whether is not True:
            out = attn(q_, k_, v_)
        else:
            mask_tile = mask.repeat_interleave(n_f, 0)
            out = attn_mask(q_, k_, v_, mask_tile)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # to out
        x = self.proj(out)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """
    CrossAttention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q_question = nn.Linear(dim, dim * 1, bias=qkv_bias)

        self.kv_video = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, question, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        # project x to q, k, v vaalues
        q = self.q_question(question)
        k, v = self.kv_video(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q *= self.scale
        cls_q = q[:, 0:1]
        q_ = q[:, 1:]

        cls_out = attn(cls_q, k, v)

        n_f = int(einops_dims['f'])
        q_ = q_.repeat_interleave(n_f, 0)
        k, v = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (k, v))

        out = attn(q_, k, v)
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)
        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        # to out
        x = self.proj(out)
        x = self.proj_drop(x)
        return x


class VideoBridgeBlock(nn.Module):
    """
    dim:
    num_heads:
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = VarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.crossattn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.bridgeattn = VarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.bridge_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm4 = norm_layer(dim)
        self.norm5 = norm_layer(dim)
        self.norm6 = norm_layer(dim)
        self.norm7 = norm_layer(dim)

    def forward(self, x_bridge, x, question, mask, layer, einops_from_space, einops_to_space, space_f):

        space_output = self.attn(self.norm1(x), mask, False, einops_from_space,
                                 einops_to_space, f=space_f)
        space_residual = x + self.drop_path(space_output)
        x_after = space_residual + self.drop_path(self.mlp(self.norm2(space_residual)))

        cross_out = self.crossattn(self.norm4(x[:, 1:]), self.norm5(question),
                                   einops_from_space, einops_to_space, f=space_f)
        if layer == 0:
            bridge_temp = cross_out
        else:
            bridge_temp = cross_out + x_bridge

        space_bridge_output = self.bridgeattn(self.norm7(bridge_temp), mask, True, einops_from_space,
                                              einops_to_space, f=space_f)
        space_bridge_residual = bridge_temp + self.drop_path(space_bridge_output)
        x_bridge_after = space_bridge_residual + self.drop_path(self.bridge_mlp(self.norm6(space_bridge_residual)))

        return x_bridge_after, x_after


class VideoBridgeFormer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `Space-Time Transformer` from Frozen-in-time  - by Max Bain.
        https://arxiv.org/abs/2104.00650

    Based off:
     - ViT implementation from the timm library [https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py]
    lucidrains timesformer implementation [https://github.com/lucidrains/TimeSformer-pytorch].

    Notable differences:
     - allows for variable length input frames (<= num_frames)
     - allows for variable length input resolution  (<= (img_size, img_size)) [UNTESTED]
     - different attention block mechanism
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 num_frames=8):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            num_frames: (int) maximum number of frames expected as input

        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed2D(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, flatten=False)

        num_patches = self.patch_embed.num_patches*num_frames
        self.patches_per_frame = num_patches // num_frames

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patches_per_frame + 1,
                        embed_dim))  # remember to take pos_embed[1:] for tiling over time

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            VideoBridgeBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # if num_frames > 1, then we perform ViT inflation and initialise time attention to zero so not necessary.
        if num_frames == 1:
            self.apply(init_vit_weights)

        # einops transformations
        self.einops_from_space = 'b (f n) d'
        self.einops_to_space = '(b f) n d'

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, question, mask):
        b, curr_frames, _, _, _ = x.shape
        x = self.patch_embed(x.transpose(2, 1))
        x = x.flatten(2).transpose(2, 1)
        x = x.reshape(b, -1, self.embed_dim)

        bf = x.shape[0]
        cls_tokens = self.cls_token.expand(bf, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
        cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
        tile_pos_embed = self.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
        total_pos_embed = tile_pos_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

        curr_patches = x.shape[1]
        x = x + total_pos_embed[:, :curr_patches]
        x = self.pos_drop(x)

        f = curr_frames

        x_bridge = x
        layer = 0
        for blk in self.blocks:
            id_ = int(layer / 2)
            question_temp = question[id_]

            x_bridge, x = blk(x_bridge, x, question_temp, mask, layer, self.einops_from_space,
                              self.einops_to_space, space_f=f)
            layer = layer + 1

        x = self.norm1(x)[:, 0]
        x = self.pre_logits(x)
        x_bridge = self.norm2(x_bridge)[:, 0]
        x_bridge = self.pre_logits(x_bridge)

        return x_bridge, x

    def forward(self, x, question, mask):
        x_bridge, x = self.forward_features(x, question, mask)
        x_bridge = self.head(x_bridge)
        x = self.head(x)
        return x_bridge, x

