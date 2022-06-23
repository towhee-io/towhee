# Copyright 2022 Zilliz. All rights reserved.
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
#
# Code inspired by https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html
# https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv3.html
#
import os
try:
    from fvcore.common.config import CfgNode as CN
except ModuleNotFoundError:
    os.system('pip install fvcore')
    from fvcore.common.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()

###################
####UniFormerS8 ###
###################
_C.MODEL.UniFormerS8 = CN()
_C.MODEL.UniFormerS8.depth = (3, 4, 8, 3)
_C.MODEL.UniFormerS8.num_classes = 400
_C.MODEL.UniFormerS8.img_size = 256
_C.MODEL.UniFormerS8.in_chans = 3
_C.MODEL.UniFormerS8.embed_dim = (64, 128, 320, 512)
_C.MODEL.UniFormerS8.head_dim = 64
_C.MODEL.UniFormerS8.mlp_ratio = 4
_C.MODEL.UniFormerS8.qkv_bias = True
_C.MODEL.UniFormerS8.qk_scale = None
_C.MODEL.UniFormerS8.representation_size = None
_C.MODEL.UniFormerS8.drop_rate = 0
_C.MODEL.UniFormerS8.attn_drop_rate = 0
_C.MODEL.UniFormerS8.drop_path_rate = 0.1
_C.MODEL.UniFormerS8.split = False
_C.MODEL.UniFormerS8.std = False
_C.MODEL.UniFormerS8.use_checkpoint = True
_C.MODEL.UniFormerS8.checkpoint_num = (0, 0, 0, 0)
_C.MODEL.UniFormerS8.pretrain_name = 'uniformer_small_k400_8x8'

###################
####UniFormerS16 ###
###################
_C.MODEL.UniFormerS16 = CN()
_C.MODEL.UniFormerS16.depth = (3, 4, 8, 3)
_C.MODEL.UniFormerS16.num_classes = 400
_C.MODEL.UniFormerS16.img_size = 256
_C.MODEL.UniFormerS16.in_chans = 3
_C.MODEL.UniFormerS16.embed_dim = (64, 128, 320, 512)
_C.MODEL.UniFormerS16.head_dim = 64
_C.MODEL.UniFormerS16.mlp_ratio = 4
_C.MODEL.UniFormerS16.qkv_bias = True
_C.MODEL.UniFormerS16.qk_scale = None
_C.MODEL.UniFormerS16.representation_size = None
_C.MODEL.UniFormerS16.drop_rate = 0
_C.MODEL.UniFormerS16.attn_drop_rate = 0
_C.MODEL.UniFormerS16.drop_path_rate = 0.1
_C.MODEL.UniFormerS16.split = False
_C.MODEL.UniFormerS16.std = False
_C.MODEL.UniFormerS16.use_checkpoint = True
_C.MODEL.UniFormerS16.checkpoint_num = (0, 0, 0, 0)
_C.MODEL.UniFormerS16.pretrain_name = 'uniformer_small_k400_16x4'

###################
####UniFormerB8 ###
###################
_C.MODEL.UniFormerB8 = CN()
_C.MODEL.UniFormerB8.depth = (5, 8, 20, 7)
_C.MODEL.UniFormerB8.num_classes = 400
_C.MODEL.UniFormerB8.img_size = 256
_C.MODEL.UniFormerB8.in_chans = 3
_C.MODEL.UniFormerB8.embed_dim = (64, 128, 320, 512)
_C.MODEL.UniFormerB8.head_dim = 64
_C.MODEL.UniFormerB8.mlp_ratio = 4
_C.MODEL.UniFormerB8.qkv_bias = True
_C.MODEL.UniFormerB8.qk_scale = None
_C.MODEL.UniFormerB8.representation_size = None
_C.MODEL.UniFormerB8.drop_rate = 0
_C.MODEL.UniFormerB8.attn_drop_rate = 0
_C.MODEL.UniFormerB8.drop_path_rate = 0.1
_C.MODEL.UniFormerB8.split = False
_C.MODEL.UniFormerB8.std = False
_C.MODEL.UniFormerB8.use_checkpoint = True
_C.MODEL.UniFormerB8.checkpoint_num = (0, 0, 0, 0)
_C.MODEL.UniFormerB8.pretrain_name = 'uniformer_base_k400_8x8'

###################
####UniFormerB16###
###################
_C.MODEL.UniFormerB16 = CN()
_C.MODEL.UniFormerB16.depth = (5, 8, 20, 7)
_C.MODEL.UniFormerB16.num_classes = 400
_C.MODEL.UniFormerB16.img_size = 256
_C.MODEL.UniFormerB16.in_chans = 3
_C.MODEL.UniFormerB16.embed_dim = (64, 128, 320, 512)
_C.MODEL.UniFormerB16.head_dim = 64
_C.MODEL.UniFormerB16.mlp_ratio = 4
_C.MODEL.UniFormerB16.qkv_bias = True
_C.MODEL.UniFormerB16.qk_scale = None
_C.MODEL.UniFormerB16.representation_size = None
_C.MODEL.UniFormerB16.drop_rate = 0
_C.MODEL.UniFormerB16.attn_drop_rate = 0
_C.MODEL.UniFormerB16.drop_path_rate = 0.1
_C.MODEL.UniFormerB16.split = False
_C.MODEL.UniFormerB16.std = False
_C.MODEL.UniFormerB16.use_checkpoint = True
_C.MODEL.UniFormerB16.checkpoint_num = (0, 0, 0, 0)
_C.MODEL.UniFormerB16.pretrain_name = 'uniformer_base_k400_16x4'
