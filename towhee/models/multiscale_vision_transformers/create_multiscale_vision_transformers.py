# Copyright 2021 Zilliz and Facebook. All rights reserved.
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
from functools import partial

from towhee.models.layers.vision_transformer_basic_head import VisionTransformerBasicHead
from towhee.models.layers.patch_embedding import PatchEmbedding
from towhee.models.layers.spatial_temporal_cls_positional_encoding import SpatialTemporalClsPositionalEncoding
from towhee.models.utils.round_width import round_width
from towhee.models.layers.muti_scale_transformer_block import MultiScaleBlock
from towhee.models.layers.multiscale_vision_transformers import MultiscaleVisionTransformers


def create_multiscale_vision_transformers(
    spatial_size,
    temporal_size,
    cls_embed_on=True,
    sep_pos_embed=True,
    depth=16,
    norm=nn.LayerNorm,
    enable_patch_embed=True,
    input_channels=3,
    patch_embed_dim=96,
    conv_patch_embed_kernel=(3, 7, 7),
    conv_patch_embed_stride=(2, 4, 4),
    conv_patch_embed_padding=(1, 3, 3),
    enable_patch_embed_norm=False,
    use_2d_patch=False,
    num_heads=1,
    mlp_ratio=4.0,
    qkv_bias=True,
    dropout_rate_block=0.0,
    droppath_rate_block=0.0,
    pooling_mode=nn.Conv3d,
    pool_first=False,
    embed_dim_mul=None,
    atten_head_mul=None,
    pool_q_stride_size=None,
    pool_kv_stride_size=None,
    pool_kv_stride_adaptive=None,
    pool_kvq_kernel=None,
    head=VisionTransformerBasicHead,
    head_dropout_rate=0.5,
    head_activation=None,
    head_num_classes=400
) -> nn.Module:
    """
    Multiscale Vision Transformers
    https://arxiv.org/abs/2104.11227
    ::
                                           PatchEmbed
                                               ↓
                                       PositionalEncoding
                                               ↓
                                            Dropout
                                               ↓
                                         Normalization
                                               ↓
                                             Block 1
                                               ↓
                                               .
                                               .
                                               .
                                               ↓
                                             Block N
                                               ↓
                                         Normalization
                                               ↓
                                              Head
    Args:
        spatial_size ('_size_2_t'):
            Input video spatial resolution (H, W). If a single int is given,
            it assumes the width and the height are the same.
        temporal_size ('int'):
            Number of frames in the input video.
        cls_embed_on ('bool'):
            If True, use cls embed in the model. Otherwise features are average
            pooled before going to the final classifier.
        sep_pos_embed ('bool'):
            If True, perform separate spatiotemporal embedding.
        depth ('int'):
            The depth of the model.
        norm ('Callable'):
            Normalization layer.
        enable_patch_embed ('bool'):
            If true, patchify the input video. If false, it assumes the input should
            have the feature dimension of patch_embed_dim.
        input_channels ('int'):
            Channel dimension of the input video.
        patch_embed_dim ('int'):
            Embedding dimension after patchifing the video input.
        conv_patch_embed_kernel ('Tuple[int]'):
            Kernel size of the convolution for patchifing the video input.
        conv_patch_embed_stride ('Tuple[int]'):
            Stride size of the convolution for patchifing the video input.
        conv_patch_embed_padding ('Tuple[int]'):
            Padding size of the convolution for patchifing the video input.
        enable_patch_embed_norm ('bool'):
            If True, apply normalization after patchifing the video input.
        use_2d_patch ('bool'):
            If True, use 2D convolutions to get patch embed. Otherwise, use 3D convolutions.
        num_heads ('int'):
            Number of heads in the first transformer block.
        mlp_ratio ('float'):
            MLP ratio which controls the feature dimension in the hidden layer of the Mlp block.
        qkv_bias ('bool'):
            If set to False, the qkv layer will not learn an additive bias.
        dropout_rate_block ('float'):
            Dropout rate for the attention block.
        droppath_rate_block ('float'):
            Droppath rate for the attention block.
        pooling_mode ('Callable'):
            Pooling mode.
        pool_first ('bool'):
            If set to True, pool is applied before qkv projection. Otherwise, pool is applied after qkv projection.
        embed_dim_mul ('List[List[int]]'):
            Dimension multiplication at layer i. If X is used, then the next block will increase
            the embed dimension by X times. Format: [depth_i, mul_dim_ratio].
        atten_head_mul ('List[List[int]]'):
            Head dimension multiplication at  layer i. If X is used, then the next block will increase
            the head by X times. Format: [depth_i, mul_dim_ratio].
        pool_q_stride_size ('List[List[int]]'):
            List of stride sizes for the pool q at each layer. Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,].
        pool_kv_stride_size ('List[List[int]]'):
            List of stride sizes for the pool kv at each layer. Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,].
        pool_kv_stride_adaptive ('_size_3_t'):
            Initial kv stride size for the first block. The stride size will be further reduced at the layer
            where q is pooled with the ratio of the stride of q pooling. If pool_kv_stride_adaptive is set,
            then pool_kv_stride_size should be none.
        pool_kvq_kernel ('_size_3_t'):
            Pooling kernel size for q and kv. It None, the kernel_size is [s + 1 if s > 1 else s for s in stride_size].
        head ('Callable'):
            Head model.
        head_dropout_rate ('float'):
            Dropout rate in the head.
        head_activation ('float'):
            Activation in the head.
        head_num_classes ('int'):
            Number of classes in the final classification head.
    """

    if use_2d_patch:
        assert temporal_size == 1, "If use_2d_patch, temporal_size needs to be 1."
    if pool_kv_stride_adaptive is not None:
        assert (
                pool_kv_stride_size is None
        ), "pool_kv_stride_size should be none if pool_kv_stride_adaptive is set."
    if norm == nn.LayerNorm:
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
    else:
        raise NotImplementedError("Only supports layernorm.")
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    conv_patch_op = nn.Conv2d if use_2d_patch else nn.Conv3d
    if enable_patch_embed:
        patch_embed = PatchEmbedding(
            in_channels=input_channels,
            out_channels=patch_embed_dim,
            conv_kernel_size=conv_patch_embed_kernel,
            conv_stride=conv_patch_embed_stride,
            conv_padding=conv_patch_embed_padding,
            conv=conv_patch_op,
        )
    else:
        patch_embed = None
    input_dims = [temporal_size, spatial_size[0], spatial_size[1]]
    if use_2d_patch:
        input_stirde = (1,) + tuple(conv_patch_embed_stride)
    else:
        input_stirde = conv_patch_embed_stride

    if enable_patch_embed:
        patch_embed_shape = [input_dims[i] // input_stirde[i] for i in range(len(input_dims))]
    else:
        patch_embed_shape = input_dims

    cls_positional_encoding = SpatialTemporalClsPositionalEncoding(
        embed_dim=patch_embed_dim,
        patch_embed_shape=patch_embed_shape,
        sep_pos_embed=sep_pos_embed,
        has_cls=cls_embed_on,
    )

    # stochastic depth decay rule
    dpr = [
        x.item() for x in torch.linspace(0, droppath_rate_block, depth)
    ]

    if dropout_rate_block > 0.0:
        pos_drop = nn.Dropout(p=dropout_rate_block)

    dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
    if embed_dim_mul is not None:
        for i in range(len(embed_dim_mul)):
            dim_mul[embed_dim_mul[i][0]] = embed_dim_mul[i][1]
    if atten_head_mul is not None:
        for i in range(len(atten_head_mul)):
            head_mul[atten_head_mul[i][0]] = atten_head_mul[i][1]

    norm_patch_embed = norm_layer(patch_embed_dim) if enable_patch_embed_norm else None

    pool_q = [[] for i in range(depth)]
    pool_kv = [[] for i in range(depth)]
    stride_q = [[] for i in range(depth)]
    stride_kv = [[] for i in range(depth)]

    if pool_q_stride_size is not None:
        for i in range(len(pool_q_stride_size)):
            stride_q[pool_q_stride_size[i][0]] = pool_q_stride_size[i][1:]
            if pool_kvq_kernel is not None:
                pool_q[pool_q_stride_size[i][0]] = pool_kvq_kernel
            else:
                pool_q[pool_q_stride_size[i][0]] = [
                    s + 1 if s > 1 else s for s in pool_q_stride_size[i][1:]
                ]

    if pool_kv_stride_adaptive is not None:
        stride_kv = pool_kv_stride_adaptive
        pool_kv_stride_size = []
        for i in range(depth):
            if len(stride_q[i]) > 0:
                stride_kv = [
                    max(stride_kv[d] // stride_q[i][d], 1)
                    for d in range(len(stride_kv))
                ]
            pool_kv_stride_size.append([i] + stride_kv)

    if pool_kv_stride_size is not None:
        for i in range(len(pool_kv_stride_size)):
            stride_kv[pool_kv_stride_size[i][0]] = pool_kv_stride_size[i][1:]
            if pool_kvq_kernel is not None:
                pool_kv[pool_kv_stride_size[i][0]] = pool_kvq_kernel
            else:
                pool_kv[pool_kv_stride_size[i][0]] = [
                    s + 1 if s > 1 else s for s in pool_kv_stride_size[i][1:]
                ]

    mvit_blocks = nn.ModuleList()
    for i in range(depth):
        num_heads = round_width(num_heads, head_mul[i], min_width=1, divisor=1)
        patch_embed_dim = round_width(patch_embed_dim, dim_mul[i], divisor=num_heads)
        dim_out = round_width(
            patch_embed_dim,
            dim_mul[i + 1],
            divisor=round_width(num_heads, head_mul[i + 1]),
        )

        mvit_blocks.append(
            MultiScaleBlock(
                dim=patch_embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate_block,
                droppath_rate=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i],
                kernel_kv=pool_kv[i],
                stride_q=stride_q[i],
                stride_kv=stride_kv[i],
                pool_mode=pooling_mode,
                has_cls_embed=cls_embed_on,
                pool_first=pool_first,
            )
        )

    embed_dim = dim_out
    norm_embed = norm_layer(embed_dim)
    if head is not None:
        head_model = head(
            in_features=embed_dim,
            out_features=head_num_classes,
            seq_pool_type="cls" if cls_embed_on else "mean",
            dropout_rate=head_dropout_rate,
            activation=head_activation,
        )
    else:
        head_model = None
    return MultiscaleVisionTransformers(
        patch_embed=patch_embed,
        cls_positional_encoding=cls_positional_encoding,
        pos_drop=pos_drop if dropout_rate_block > 0.0 else None,
        norm_patch_embed=norm_patch_embed,
        blocks=mvit_blocks,
        norm_embed=norm_embed,
        head=head_model,
    )
