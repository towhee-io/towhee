# Original pytorch implementation by:
# 'Wave-ViT: Unifying Wavelet and Transformers for Visual Representation Learning'
#       - http://arxiv.org/pdf/2207.04978
# Original code by / Copyright 2022, YehLi.
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
import torch
from torch import nn
from typing import List, Any
import numpy as np
from timm.models.vision_transformer import _cfg
from towhee.models.wave_vit.wave_vit_block import rand_bbox, Stem, DownSamples, Block, ClassBlock
from towhee.models.utils.init_vit_weights import init_vit_weights
from towhee.models.wave_vit.wave_vit_utils import get_configs


class WaveViT(nn.Module):
    """
    WaveViT
    Args:
        in_chans (`int`):
            Number of image channels.
        num_classes (`int`):
            Number of image classes.
        stem_hidden_dim (`int`):
            Stem layer hidden dim.
        embed_dims (`list`):
            Every layer`s embedding dim.
        num_heads (`list`):
            Every layer`s head number.
        mlp_ratios (`list`):
            Every layer`s mlp ration.
        drop_path_rate (`float`):
            Every layer`s drop rate.
        norm_layer (`nn.Module`):
            Default nn.LayerNorm.
        depths (`int`):
            Number of depths.
        sr_ratios (`list`):
            Every block`s sr ration.
        num_stages (`int`):
            Number of stages.
        token_label (`bool`):
            Whether to use the token_label.
    """
    def __init__(self,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 stem_hidden_dim: int = 32,
                 embed_dims: List = None,
                 num_heads: List = None,
                 mlp_ratios: List = None,
                 drop_path_rate: float = 0.,
                 norm_layer: Any = nn.LayerNorm,
                 depths: int = None,
                 sr_ratios: List = None,
                 num_stages: int = 4,
                 token_label: bool = True,
                 ):
        super().__init__()
        if embed_dims is None:
            embed_dims = [64, 128, 320, 448]
        if num_heads is None:
            num_heads = [2, 4, 10, 14]
        if mlp_ratios is None:
            mlp_ratios = [8, 8, 4, 4]
        if depths is None:
            depths = [3, 4, 6, 3]
        if sr_ratios is None:
            sr_ratios = [4, 2, 1, 1]
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = Stem(in_chans, stem_hidden_dim, embed_dims[i])
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                drop_path=dpr[cur + j],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[i],
                block_type="wave" if i < 2 else "std_att")
                for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        post_layers = ["ca"]
        self.post_network = nn.ModuleList([
            ClassBlock(
                dim=embed_dims[-1],
                num_heads=num_heads[-1],
                mlp_ratio=mlp_ratios[-1],
                norm_layer=norm_layer)
            for _ in range(len(post_layers))
        ])

        # classification head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        # token_label
        self.return_dense = token_label
        self.mix_token = token_label
        self.beta = 1.0
        self.pooling_scale = 8
        if self.return_dense:
            self.aux_head = nn.Linear(
                embed_dims[-1],
                num_classes) if num_classes > 0 else nn.Identity()
        # token_label

        self.apply(init_vit_weights)

    def forward_cls(self, x):
        _, _, _ = x.shape
        cls_tokens = x.mean(dim=1, keepdim=True)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.post_network:
            x = block(x)
        return x

    def forward_features(self, x):
        b = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, h, w = patch_embed(x)

            for blk in block:

                x = blk(x, h, w)

            if i != self.num_stages - 1:
                norm = getattr(self, f"norm{i + 1}")
                x = norm(x)
                x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        x = self.forward_cls(x)[:, 0]
        norm = getattr(self, f"norm{self.num_stages}")
        x = norm(x)
        return x

    def forward(self, x):
        if not self.return_dense:
            x = self.forward_features(x)
            x = self.head(x)
            return x
        else:
            x, h, w = self.forward_embeddings(x)
            # mix token, see token labeling for details.
            if self.mix_token and self.training:
                lam = np.random.beta(self.beta, self.beta)
                patch_h, patch_w = x.shape[1] // self.pooling_scale, x.shape[
                    2] // self.pooling_scale
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam, scale=self.pooling_scale)
                temp_x = x.clone()
                sbbx1, sbby1, sbbx2, sbby2 = self.pooling_scale * bbx1, self.pooling_scale * bby1, \
                                             self.pooling_scale * bbx2, self.pooling_scale * bby2
                temp_x[:, sbbx1:sbbx2, sbby1:sbby2, :] = x.flip(0)[:, sbbx1:sbbx2, sbby1:sbby2, :]
                x = temp_x
            else:
                bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0

            x = self.forward_tokens(x, h, w)
            x_cls = self.head(x[:, 0])
            x_aux = self.aux_head(
                x[:, 1:]
            )  # generate classes in all feature tokens, see token labeling

            if not self.training:
                return x_cls + 0.5 * x_aux.max(1)[0]

            if self.mix_token and self.training:  # reverse "mix token", see token labeling for details.
                x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])

                temp_x = x_aux.clone()
                temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
                x_aux = temp_x

                x_aux = x_aux.reshape(x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1])

            return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)

    def forward_tokens(self, x, h, w):
        b = x.shape[0]
        x = x.view(b, -1, x.size(-1))

        for i in range(self.num_stages):
            if i != 0:
                patch_embed = getattr(self, f"patch_embed{i + 1}")
                x, h, w = patch_embed(x)

            block = getattr(self, f"block{i + 1}")
            for blk in block:
                x = blk(x, h, w)

            if i != self.num_stages - 1:
                norm = getattr(self, f"norm{i + 1}")
                x = norm(x)
                x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        x = self.forward_cls(x)
        norm = getattr(self, f"norm{self.num_stages}")
        x = norm(x)
        return x

    def forward_embeddings(self, x):
        patch_embed = getattr(self, f"patch_embed{0 + 1}")
        x, h, w = patch_embed(x)
        x = x.view(x.size(0), h, w, -1)
        return x, h, w


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
        model = WaveViT(**kwargs)
    else:
        configs = get_configs(model_name)
        configs.update(**kwargs)
        model = WaveViT(**configs)
        model.default_cfg = _cfg()
        if pretrained:
            state_dic = torch.load(weights_path, map_location=device)["state_dict"]
            model.load_state_dict(state_dic)
    # add .float() to solve the following problems
    # Input type (torch.FloatTensor) and weight type (torch.HalfTensor) should be the same
    model.to(device).float()
    model.eval()

    return model


# if __name__ == '__main__':
#     path1 = '/Users/zilliz/PycharmProjects/pretrain/Wave-ViT/wavevit_s.pth.tar'
#     path2 = '/Users/zilliz/PycharmProjects/pretrain/Wave-ViT/wavevit_s_384.pth.tar'
#     model = create_model(model_name='wave_vit_s', pretrained=False, weights_path=path1,
#                          token_label=False,)
#     query_image = torch.randn(1, 3, 224, 224)
#     out = model(query_image)
#
#     print(out.shape)



