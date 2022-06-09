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
#### TSMK400R50 ###
###################

_C.MODEL.TSMK400R50 = CN()
_C.MODEL.TSMK400R50.weights = 'TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth'
_C.MODEL.TSMK400R50.num_class = 400
_C.MODEL.TSMK400R50.is_shift = True
_C.MODEL.TSMK400R50.num_segments = 8
_C.MODEL.TSMK400R50.new_length = 1
_C.MODEL.TSMK400R50.input_modality = 'RGB'
_C.MODEL.TSMK400R50.consensus_module = 'avg'
_C.MODEL.TSMK400R50.dropout_ratio = 0.8
_C.MODEL.TSMK400R50.img_feature_dim = 256
_C.MODEL.TSMK400R50.base_model = 'resnet50'
_C.MODEL.TSMK400R50.pretrain = False
_C.MODEL.TSMK400R50.shift_place = 'blockres'
_C.MODEL.TSMK400R50.shift_div = 8
_C.MODEL.TSMK400R50.non_local = False

###################
#### TSMSOMER50 ###
###################

_C.MODEL.TSMSOMER50 = CN()
_C.MODEL.TSMSOMER50.weights = 'TSM_something_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth'
_C.MODEL.TSMSOMER50.num_class = 174
_C.MODEL.TSMSOMER50.is_shift = True
_C.MODEL.TSMSOMER50.num_segments = 8
_C.MODEL.TSMSOMER50.new_length = 1
_C.MODEL.TSMSOMER50.input_modality = 'RGB'
_C.MODEL.TSMSOMER50.consensus_module = 'avg'
_C.MODEL.TSMSOMER50.dropout_ratio = 0.8
_C.MODEL.TSMSOMER50.img_feature_dim = 256
_C.MODEL.TSMSOMER50.base_model = 'resnet50'
_C.MODEL.TSMSOMER50.pretrain = False
_C.MODEL.TSMSOMER50.shift_place = 'blockres'
_C.MODEL.TSMSOMER50.shift_div = 8
_C.MODEL.TSMSOMER50.non_local = False

###################
#TSMSOMEV2R50SEG16#
###################

_C.MODEL.TSMSOMEV2R50SEG16 = CN()
_C.MODEL.TSMSOMEV2R50SEG16.weights = 'TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment16_e45.pth'
_C.MODEL.TSMSOMEV2R50SEG16.num_class = 174
_C.MODEL.TSMSOMEV2R50SEG16.is_shift = True
_C.MODEL.TSMSOMEV2R50SEG16.num_segments = 16
_C.MODEL.TSMSOMEV2R50SEG16.new_length = 1
_C.MODEL.TSMSOMEV2R50SEG16.input_modality = 'RGB'
_C.MODEL.TSMSOMEV2R50SEG16.consensus_module = 'avg'
_C.MODEL.TSMSOMEV2R50SEG16.dropout_ratio = 0.8
_C.MODEL.TSMSOMEV2R50SEG16.img_feature_dim = 256
_C.MODEL.TSMSOMEV2R50SEG16.base_model = 'resnet50'
_C.MODEL.TSMSOMEV2R50SEG16.pretrain = False
_C.MODEL.TSMSOMEV2R50SEG16.shift_place = 'blockres'
_C.MODEL.TSMSOMEV2R50SEG16.shift_div = 8
_C.MODEL.TSMSOMEV2R50SEG16.non_local = False

###################
##TSMFlowK400R50 ##
###################

_C.MODEL.TSMFlowK400R50 = CN()
_C.MODEL.TSMFlowK400R50.weights = 'TSM_kinetics_Flow_resnet50_shift8_blockres_avg_segment8_e50.pth'
_C.MODEL.TSMFlowK400R50.num_class = 400
_C.MODEL.TSMFlowK400R50.is_shift = True
_C.MODEL.TSMFlowK400R50.num_segments = 8
_C.MODEL.TSMFlowK400R50.new_length = 5
_C.MODEL.TSMFlowK400R50.input_modality = 'RGB'
_C.MODEL.TSMFlowK400R50.consensus_module = 'avg'
_C.MODEL.TSMFlowK400R50.dropout_ratio = 0.8
_C.MODEL.TSMFlowK400R50.img_feature_dim = 256
_C.MODEL.TSMFlowK400R50.base_model = 'resnet50'
_C.MODEL.TSMFlowK400R50.pretrain = False
_C.MODEL.TSMFlowK400R50.shift_place = 'blockres'
_C.MODEL.TSMFlowK400R50.shift_div = 8
_C.MODEL.TSMFlowK400R50.non_local = False
