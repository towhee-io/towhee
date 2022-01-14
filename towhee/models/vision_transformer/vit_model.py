import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict

from towhee.models.layers.vit_block import Block
from towhee.models.layers.patch_embed import PatchEmbed

def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class VitModel(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                 representation_size=None, drop_ratio=0, attn_drop_ratio=0, drop_path_ratio=0,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        super(VitModel, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        depth_decay = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                drop_path_ratio=drop_path_ratio,
                norm_layer=self.norm_layer,
                act_layer=self.act_layer
            )
            for i in range(depth)
        ])
        self.norm = self.norm_layer(embed_dim)

        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def vit_base_16_224(num_classes: int=1000):
    model = VitModel(
        img_size=224, patch_size=16, num_classes=num_classes,
        embed_dim=768, depth=12, num_heads=12,
        representation_size=None,
    )
    return model

def vit_base_32_224(num_classes: int=1000):
    model = VitModel(
        img_size=224, patch_size=32, num_classes=num_classes,
        embed_dim=768, depth=12, num_heads=12,
        representation_size=None,
    )
    return model

def vit_large_16_224(num_classes: int=1000):
    model = VitModel(
        img_size=224, patch_size=16, num_classes=num_classes,
        embed_dim=1024, depth=24, num_heads=16,
        representation_size=None,
    )
    return model

def vit_large_32_224(num_classes: int=1000):
    model = VitModel(
        img_size=224, patch_size=32, num_classes=num_classes,
        embed_dim=1024, depth=24, num_heads=16,
        representation_size=None,
    )
    return model