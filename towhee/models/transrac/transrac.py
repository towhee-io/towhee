# Implementation of TransRAC in paper:
#   [TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting]
#   (https://arxiv.org/abs/2204.01018)
#
# Inspired by official code from https://github.com/SvipRepetitionCounting/TransRAC
#
# Modifications & additions by Copyright 2021 Zilliz. All rights reserved.
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

from towhee.models.layers.transformer_encoder import TransformerEncoder
from towhee.models.transrac.utils import DenseMap, SimilarityMatrix


class TransRAC(nn.Module):
    """
    TransRAC model

    Args:
        backbone (`str`):
            Model name of pretrained backbone.
        num_frames (`int`):
            Number of video frames.
        scales (`list`):
            A list of scale numbers.

    Example:
        >>> import torch
        >>> from towhee.models import video_swin_transformer
        >>> from towhee.models.transrac import TransRAC
        >>>
        >>> dummy_video = torch.rand(1, 3, 4, 200, 200) # (bcthw)
        >>>
        >>> backbone = video_swin_transformer.create_model() # use default configs here
        >>> model = TransRAC(backbone=backbone, num_frames=4)
        >>>
        >>> out, out_matrix = model(dummy_video)
        >>> print(out.shape, out_matrix.shape)
        torch.Size([1, 4]), torch.Size([1, 12, 4, 4])
    """

    def __init__(self, backbone: str, num_frames: int = 64, scales: list = None):
        super().__init__()
        if scales is None:
            scales = [1, 4, 8]
        self.num_frames = num_frames
        # self.config = config
        self.scales = scales

        # Load pretrained Video Swin Transformer
        # self.backbone = create_model(model_name=backbone, pretrained=True)
        self.backbone = backbone

        self.replication_padding1 = nn.ConstantPad3d((0, 0, 0, 0, 1, 1), 0)
        self.replication_padding2 = nn.ConstantPad3d((0, 0, 0, 0, 2, 2), 0)
        self.replication_padding4 = nn.ConstantPad3d((0, 0, 0, 0, 4, 4), 0)

        self.conv3d = nn.Conv3d(in_channels=768,
                                out_channels=512,
                                kernel_size=3,
                                padding=(3, 1, 1),
                                dilation=(3, 1, 1))

        self.bn1 = nn.BatchNorm3d(512)
        self.spatial_pooling = nn.MaxPool3d(kernel_size=(1, 7, 7))

        self.sims = SimilarityMatrix()
        self.conv3x3 = nn.Conv2d(in_channels=4 * len(self.scales),  # num_head*scale_num
                                 out_channels=32,
                                 kernel_size=3,
                                 padding=1)

        self.bn2 = nn.BatchNorm2d(32)

        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.num_frames * 32, 512)
        self.ln1 = nn.LayerNorm(512)

        self.trans_encoder = TransformerEncoder(
            d_model=512, n_head=4, dim_ff=512, num_layers=1,
            num_frames=self.num_frames, dropout=0.2)
        self.fc = DenseMap(512, 512, 256, 1)

    def forward(self, x):
        # bcthw
        _, _, t, _, _ = x.shape
        assert t == self.num_frames

        multi_scales = []
        for scale in self.scales:
            if scale == 4:
                x = self.replication_padding2(x)
                crops = [x[:, :, i:i + scale, :, :] for i in
                         range(0, self.num_frames - scale + scale // 2 * 2, max(scale // 2, 1))]
            elif scale == 8:
                x = self.replication_padding4(x)
                crops = [x[:, :, i:i + scale, :, :] for i in
                         range(0, self.num_frames - scale + scale // 2 * 2, max(scale // 2, 1))]
            else:
                crops = [x[:, :, i:i + 1, :, :] for i in range(0, self.num_frames)]

            slices = []
            # feature extract with video SwinTransformer
            for crop in crops:
                crop = self.backbone(crop)
                slices.append(crop)

            x_scale = torch.cat(slices, dim=2)  # -> (b, 768, f, size, size)
            x_scale = nn.functional.relu(self.bn1(self.conv3d(x_scale)))  # -> (b, 512, f, 7, 7)
            x_scale = self.spatial_pooling(x_scale)  # -> (b, 512, f, 1, 1)
            x_scale = x_scale.squeeze(3).squeeze(3)  # -> (b, 512, f)
            x_scale = x_scale.transpose(1, 2)  # -> (b, f, 512)

            # similarity matrix
            x_sims = nn.functional.relu(self.sims(x_scale, x_scale, x_scale))  # -> (b, 4, f, f)
            multi_scales.append(x_sims)

        x = torch.cat(multi_scales, dim=1)  # -> (B, 4*scale_num, f, f)
        x_matrix = x  # to return matrix
        x = nn.functional.relu(self.bn2(self.conv3x3(x)))  # -> (b, 32, f, f)
        x = self.dropout1(x)

        x = x.permute(0, 2, 3, 1)  # -> (b, f, f, 32)
        x = x.flatten(start_dim=2)  # -> (b, f, 32*f)
        x = nn.functional.relu(self.input_projection(x))  # -> (b, f, 512)
        x = self.ln1(x)

        x = x.transpose(0, 1)  # -> (f, b, 512)
        x = self.trans_encoder(x)
        x = x.transpose(0, 1)  # -> (b, f, 512)

        x = self.fc(x)  # -> (b, f, 1)
        x = x.squeeze(2)

        return x, x_matrix
