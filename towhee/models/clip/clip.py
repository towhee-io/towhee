# Built on top of the original implementation at https://github.com/openai/CLIP
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

import os
import warnings
from collections import OrderedDict
from typing import Tuple, Union, Callable

import numpy as np
import torch
from torch import nn

from .clip_utils import get_configs, _download, convert_weights, patch_device, patch_float, tokenize
from towhee.models.clip.auxilary import multi_head_attention_forward, MultiheadAttention

warnings.filterwarnings("ignore", category=UserWarning)


class Bottleneck(nn.Module):
    """
    BottleNeck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    """
    Attention module for modified ResNet
    Args:
        spacial_dim (int): spatial dimension
        embed_dim (int): embedding dimension
        num_heads (int): number of heads
        output_dim (int): output dimension
    """

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, vis=False):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.vis = vis

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        multi_head_attention_forward_func = nn.functional.multi_head_attention_forward
        if self.vis:
            multi_head_attention_forward_func = multi_head_attention_forward
        x, _ = multi_head_attention_forward_func(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, vis=False):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim, vis)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x):  # pylint: disable=W0237
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """
    QuickGELU
    """

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    ResidualAttentuonBlock
    """

    def __init__(self, d_model: int, n_head: int,
                 attn_mask: Union[torch.Tensor, Callable] = None, vis=False, patch_nums=None,
                 is_bridge_former_video=False):
        super().__init__()
        self.vis = vis
        self.attn = nn.MultiheadAttention(d_model, n_head)
        if vis:
            self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.patch_nums = patch_nums
        self.is_bridge_former_video = is_bridge_former_video
        self.attn_probs = None
        self.attn_grad = None

    def set_attn_probs(self, attn_probs):
        self.attn_probs = attn_probs

    def set_attn_grad(self, attn_grad):
        self.attn_grad = attn_grad

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, "__call__"):
            attn_mask_ = self.attn_mask(x.size(0))  # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        if self.vis:
            return \
            self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_, attention_probs_forward_hook=self.set_attn_probs,
                      attention_probs_backwards_hook=self.set_attn_grad)[0]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def attention_frames(self, x: torch.Tensor):
        self.attn_mask = None
        bz = x.shape[1]
        # print(x.shape)
        cls_x = x[0:1,:]
        cls_out = self.attn(cls_x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        x_ = x[1:,:].permute(1, 0, 2)

        x_ = x_.reshape(-1, self.patch_nums, x_.shape[-1])
        n_f = int(x_.shape[0] / bz)  # num frames
        cls_x_tile = cls_x.permute(1, 0, 2).repeat_interleave(n_f,0)
        cls_x_cat = torch.cat([cls_x_tile,x_],1)
        x_ = x_.permute(1, 0, 2)
        cls_x_cat = cls_x_cat.permute(1, 0, 2)
        out_ = self.attn(x_, cls_x_cat, cls_x_cat, need_weights=False, attn_mask=self.attn_mask)[0]
        out_ = out_.permute(1, 0, 2)
        out_ = out_.reshape(bz, -1, out_.shape[-1])
        out_ = out_.permute(1, 0, 2)
        out = torch.cat([cls_out,out_],0)
        return out

    def forward(self, x: torch.Tensor):

        ## text transformer or visual transformer for a single frame
        if not self.is_bridge_former_video:
            x = x + self.attention(self.ln_1(x))
        ## visual transformer for multiple frames
        else:
            x = x + self.attention_frames(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """
    Transformer
    """
    def __init__(self, width: int, layers: int, heads: int,
                 attn_mask: Union[torch.Tensor, Callable] = None, vis: bool = False,
                 patch_nums: int = None, is_bridge_former_video: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(
            width, heads, attn_mask, vis,
            patch_nums=patch_nums, is_bridge_former_video=is_bridge_former_video) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    """
    ViT
    """

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, vis: bool = False, is_bridgeformer: bool = False,
                 is_bridge_former_video: bool = False):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.is_bridgeformer = is_bridgeformer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.patch_nums = (input_resolution // patch_size) ** 2
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.patch_nums+1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, vis=vis, patch_nums=self.patch_nums,
                                       is_bridge_former_video=is_bridge_former_video)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        if self.is_bridgeformer:
            bz = x.shape[0]
            n_frames = x.shape[1]
            c = x.shape[2]
            h = x.shape[3]
            w = x.shape[4]
            x = x.contiguous().view(-1, c, h, w)

            x = self.conv1(x)  # shape = [bz*n_frames, width, grid, grid]

            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [bz*n_frames, width, grid*grid]

            x = x.permute(0, 2, 1)  # shape = [bz*n_frames, grid*grid, width]

            x = x.reshape(bz, -1, x.shape[-1])  # shape = [bz, n_frames*grid*grid, width]

            cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                 device=x.device)   # shape = [bz, 1, width]

            x = torch.cat([cls, x], dim=1)  # shape = [bz, n_frames*grid*grid + 1, width]
            cls_embed = self.positional_embedding[0:1, :]  # shape = [1, width]
            tile_pos_embed = self.positional_embedding[1:, :].repeat(n_frames, 1)  # shape = [n_frames*grid*grid, width]
            # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
            total_pos_embed = torch.cat([cls_embed, tile_pos_embed], dim=0)  # shape = [n_frames*grid*grid+1, width]
            x = x + total_pos_embed.to(x.dtype)  # shape = [bz,n_frames*grid*grid+1, width]
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

            x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    """
    CLIP model
    """
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 multilingual_model: str = None,
                 context_length: int = 77,
                 vocab_size: int = 49408,
                 transformer_width: int = 512,
                 transformer_heads: int = 8,
                 transformer_layers: int = 12,
                 # whether used for CLIP4Clip model
                 clip4clip: bool = False,
                 # whether be able to visualize
                 vis: bool = False,
                 # whether is the BridgeFormer model
                 is_bridge_former: bool = False,
                 is_bridge_former_video: bool = False
                 ):
        super().__init__()

        self.multilingual_model = multilingual_model
        self.context_length = context_length
        self.is_bridge_former = is_bridge_former
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                vis=vis
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                vis=vis,
                is_bridgeformer=self.is_bridge_former,
                is_bridge_former_video=is_bridge_former_video
            )
        if clip4clip:
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask_for_clip4clip
            )
        else:
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                vis=vis
            )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_attention_mask_for_clip4clip(self, context_length):
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text, clip4clip=False, return_hidden=False, multilingual=False, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if multilingual:
            assert self.multilingual_model is not None, "Multilingual is not supported yet."
            assert isinstance(text[0], str), "Multilingual is only supported for inputs in text or list of texts."
            try:
                from multilingual_clip import pt_multilingual_clip  # pylint: disable=C0415
            except ModuleNotFoundError:
                os.system("pip install multilingual-clip")
            try:
                import transformers  # pylint: disable=C0415
            except ModuleNotFoundError:
                os.system("pip install transformers")

            tokenizer = transformers.AutoTokenizer.from_pretrained(self.multilingual_model)
            encoder = pt_multilingual_clip.MultilingualCLIP.from_pretrained(self.multilingual_model)
            x = encoder(text, tokenizer)
            return x
        else:
            if isinstance(text[0], str):
                text = tokenize(text).to(device)
            else:
                text = text.to(device)
            if clip4clip:
                x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

                pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)
                x = x + pos_emd
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD

                hidden = self.ln_final(x).type(self.dtype) @ self.text_projection

                # x.shape = [batch_size, n_ctx, transformer.width]
                # take features from the eot embedding (eot_token is the highest number in each sequence)
                x = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]

                if return_hidden:
                    return x, hidden
            else:
                x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

                x = x + self.positional_embedding.type(self.dtype)
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.ln_final(x).type(self.dtype)

                x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

            return x

    def forward(self, image, text, multilingual=False, device=None):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, multilingual=multilingual, device=device)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def create_model(
        model_name: str = None,
        pretrained: bool = False,
        weights_path: str = None,
        jit: bool = False,
        device: str = None,
        **kwargs
) -> CLIP:
    """
    Create a CLIP model.
    Args:
        model_name (`str`):
            CLIP model name, can be one of 'clip_resnet_r50', 'clip_resnet_r101', 'clip_vit_b16', 'clip_vit_b32'
        pretrained (`bool`):
            Whether to load pretrained weights.
        weights_path (`str`):
            Path to the weights file.
        jit (`bool`):
            Whether returned one is a jit model, only useful when `pretrained` is True.
        device (`str`):
            Model device to use.
        **kwargs (`dict`):
            Extra arguments to pass to the model.

    Returns:
        model (`CLIP`):
            The CLIP model.

    >>> from towhee.models import clip
    >>> model = clip.create_model("clip_resnet_r50")
    >>> model.__class__.__name__
    'CLIP'
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name is None:
        if pretrained:
            raise AttributeError("Fail to load pretrained model: no model name is specified.")
        model = CLIP(**kwargs).to(device)
    else:
        configs = get_configs(model_name)
        configs.update(**kwargs)
        if "url" in configs:
            url = configs["url"]
            configs.pop("url")
        model = CLIP(**configs).to(device)

        if pretrained:
            if weights_path:
                local_path = weights_path
            elif url:
                cache_dir = os.path.expanduser("~/.cache/clip")
                local_path = _download(url, cache_dir)
            else:
                raise AttributeError("No url or local path is provided for pretrained model.")

            try:
                try:
                    import torchvision  # pylint: disable=unused-import, import-outside-toplevel
                except ModuleNotFoundError:
                    warnings.warn("Additional package is required for jit: torchvision")

                # loading JIT archive
                model = torch.jit.load(local_path, map_location=device).eval()
                state_dict = model.state_dict()
            except RuntimeError:
                # loading saved state dict
                if jit:
                    warnings.warn(f"File {local_path} is not a JIT archive. Loading as a state dict instead")
                    jit = False
                state_dict = torch.load(local_path, map_location="cpu")

            if not jit:
                clip_model = CLIP(**configs).to(device)
                for key in ["input_resolution", "context_length", "vocab_size"]:
                    if key in state_dict:
                        del state_dict[key]

                convert_weights(model)
                clip_model.load_state_dict(state_dict)
                model = clip_model
                model.eval()
                if str(device) == "cpu":
                    model.float()
            else:
                patch_device(model, device)
                if device == "cpu":
                    patch_float(model)
    return model
