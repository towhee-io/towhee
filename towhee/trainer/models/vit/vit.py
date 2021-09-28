# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pytorch impletation of 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'.
Jax version: https://github.com/google-research/vision_transformer.
"""

import torch
from torch import nn

from towhee.trainer.models.vit.transformer import Transformer
from towhee.trainer.models.vit.vit_training_utils import load_pretrained_weights, as_tuple
from towhee.trainer.models.vit.vit_pretrained import PRETRAINED_MODELS


class PositionalEmbedding1D(nn.Module):
    """
    Adds (optionally learned) positional embeddings to the inputs.
    """

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """
        Input has shape (batch_size, seq_len, emb_dim)
        """
        return x + self.pos_embedding


class ViT(nn.Module):
    """
    Args:
        name(str, optional):
            Model name.
        weights_path(str):
            Local path to weights file.
        pretrained(bool):
            Load pretrained weights.
        patches(int):
            Number of image patches.
        dim(int):
            Number of self dimention.
        ff_dim(int):
            Number of layer size in feedforward network.
        num_heads(int):
            Number of heads.
        num_layers(int):
            Number of layers.
        attention_dropout_rate(float):
            Attention dropout rate.
        dropout_rate(float):
            Dropout rate.
        representation_size(int, optional):
            Enable and set representation layer to this value if set.
        classifier(str):
            classifier type.
        positional_embedding(str):
            positional embedding.
        in_channels(int):
            Number of input channels.
        image_size(int, optional):
            image size.
        num_classes(int, optional):
            Number of classes for classification head.
        resize_positional_embedding(bool,optional):
            If resize_positional_embedding is true.
        to_vec(bool, optional):
            If the embedding vector is predicted instead.
    """

    def __init__(
        self,
        name=None,
        weights_path=None,
        pretrained=False,
        patches=16,
        dim=768,
        ff_dim=3072,
        num_heads=12,
        num_layers=12,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        representation_size=None,
        load_repr_layer=False,
        classifier='token',
        positional_embedding='1d',
        in_channels=3,
        image_size=None,
        num_classes=None,
        resize_positional_embedding=True,
        to_vec=False
    ):
        super().__init__()
        print(f'attention_dropout_rate is {attention_dropout_rate}')
        # Configuration
        if name is None:
            check_msg = 'must specify name of pretrained model'
            assert not pretrained, check_msg
            assert not resize_positional_embedding, check_msg
            if num_classes is None:
                num_classes = 1000
            if image_size is None:
                image_size = 384
        else:  # load pretrained model
            assert name in PRETRAINED_MODELS, \
                'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            config = PRETRAINED_MODELS[name]['config']
            patches = config['patches']
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            attention_dropout_rate = config['attention_dropout_rate']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']
            classifier = config['classifier']
            if image_size is None:
                image_size = PRETRAINED_MODELS[name]['image_size']
            if num_classes is None:
                num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.image_size = image_size

        self.to_vec = to_vec

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1

        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()

        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads,
                                       ff_dim=ff_dim, dropout=dropout_rate)

        # Representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim

        # Classifier head
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        self.fc = nn.Linear(pre_logits_size, num_classes)

        # Initialize weights
        self.init_weights()

        # Load pretrained model
        if pretrained:
            pretrained_num_channels = 3
            pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
            pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
            load_pretrained_weights(
                self, name,
                weights_path=weights_path,
                load_first_conv=(in_channels == pretrained_num_channels),
                load_fc=(num_classes == pretrained_num_classes),
                load_repr_layer=load_repr_layer,
                resize_positional_embedding=(image_size != pretrained_image_size),
            )

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        """
        Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor):
                b,c,fh,fw
        """
        b, _, _, _ = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)  # b,gh*gw+1,d
        x = self.transformer(x)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            x = self.norm(x)[:, 0]  # b,d
            if self.to_vec:
                return x
            x = self.fc(x)  # b,num_classes
        return x
