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
from abc import ABC

import torch
from torch import nn
from torch.autograd import Function
import pywt
from functools import partial


def get_configs(model_name="wave_vit_s"):
    args = {
        "wave_vit_s":
            dict(stem_hidden_dim=32, embed_dims=[64, 128, 320, 448],
                 num_heads=[2, 4, 10, 14], mlp_ratios=[8, 8, 4, 4],
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], sr_ratios=[4, 2, 1, 1]),
        "wave_vit_b":
            dict(stem_hidden_dim=64, embed_dims=[64, 128, 320, 512],
                 num_heads=[2, 4, 10, 16], mlp_ratios=[8, 8, 4, 4],
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 12, 3], sr_ratios=[4, 2, 1, 1]),
        "wave_vit_l":
            dict(stem_hidden_dim=64, embed_dims=[96, 192, 384, 512],
                 num_heads=[3, 6, 12, 16], mlp_ratios=[8, 8, 4, 4],
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 6, 18, 3], sr_ratios=[4, 2, 1, 1]),
    }
    return args[model_name]


class DWTFunction(Function):
    """
    DWTFunction
    """
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]

        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            b, c, h, w = ctx.shape
            dx = dx.view(b, 4, -1, h//2, w//2)

            dx = dx.transpose(1, 2).reshape(b, -1, h//2, w//2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(c, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=c)

        return dx, None, None, None, None


class IDWTFunction(Function):
    """
    IDWTFunction
    """
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        b, _, h, w = x.shape
        x = x.view(b, 4, -1, h, w).transpose(1, 2)
        c = x.shape[1]
        x = x.reshape(b, -1, h, w)
        filters = filters.repeat(c, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=c)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            _, c, _, _ = ctx.shape
            c = c // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(c, -1, -1, -1), stride=2, groups=c)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(c, -1, -1, -1), stride=2, groups=c)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(c, -1, -1, -1), stride=2, groups=c)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(c, -1, -1, -1), stride=2, groups=c)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT2D(nn.Module):
    """
    IDWT2D
    """
    def __init__(self, wave):
        super().__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer("filters", filters)
        self.filters = self.filters.to(dtype=torch.float16)

    def forward(self, x):
        return IDWTFunction.apply(x, self.filters)


class DWT2D(nn.Module):
    """
    DWT2D
    """
    def __init__(self, wave):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer("w_ll", w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_lh", w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_hl", w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_hh", w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float16)
        self.w_lh = self.w_lh.to(dtype=torch.float16)
        self.w_hl = self.w_hl.to(dtype=torch.float16)
        self.w_hh = self.w_hh.to(dtype=torch.float16)

    def forward(self, x):
        return DWTFunction.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

