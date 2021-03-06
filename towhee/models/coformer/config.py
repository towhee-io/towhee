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
# Original code from https://github.com/jhcho99/CoFormer.
#
# Modified by Zilliz.

import os
try:
    from fvcore.common.config import CfgNode as CN
except ModuleNotFoundError:
    os.system('pip install fvcore')
    from fvcore.common.config import CfgNode as CN

vidx_ridx = [[173, 166, 167, 126], [5, 94, 52, 126], [5, 172, 126], [5, 31, 176, 168, 126],
[5, 159, 176, 126], [5, 31, 126], [5, 126], [5, 126], [5, 183, 176, 126],
[5, 1, 176, 126], [5, 176, 159, 126], [5, 179, 126], [5, 94, 170, 176, 126],
[5, 182, 126], [5, 85, 176, 126], [5, 174, 39, 126], [5, 176, 111, 126],
[5, 127, 52, 126], [5, 93, 126], [5, 94, 176, 126], [5, 94, 126], [5, 126],
[5, 111, 168, 126], [5, 111, 39, 126], [5, 94, 126], [5, 94, 176, 126], [5, 94, 176, 126],
[5, 29, 152, 126], [5, 111, 100, 176, 126], [5, 94, 159, 52, 126], [5, 126], [21, 126],
[5, 39, 176, 126], [5, 94, 176, 126], [5, 94, 126], [5, 126], [5, 94, 6, 126],
[5, 94, 100, 126], [5, 126], [8, 126], [5, 172, 126], [5, 82, 126], [5, 62, 161, 176, 126],
[5, 126], [5, 27, 126], [5, 168, 79, 126], [5, 56, 100, 176, 126], [5, 115, 52, 126],
[5, 126], [5, 126], [5, 159, 176, 126], [5, 100, 39, 126], [5, 31, 126], [5, 1, 126],
[5, 119, 6, 126], [5, 6, 1, 126], [5, 183, 18, 126], [5, 183, 185, 126], [15, 14, 176, 126],
[5, 159, 176, 52, 126], [5, 126], [5, 143, 159, 176, 126], [5, 94, 39, 126], [5, 182, 126],
[5, 166, 126], [5, 126], [5, 104, 176, 126], [5, 72, 39, 176, 126], [5, 94, 126], [5, 79, 34, 176, 126],
[5, 183, 184, 126], [5, 126], [5, 25, 176, 126], [5, 94, 126], [5, 94, 126], [5, 172, 126],
[5, 126], [5, 94, 126], [5, 94, 176, 126], [5, 94, 159, 126], [5, 157, 170, 52, 126],
[5, 168, 159, 52, 126], [5, 126], [5, 121, 126], [5, 6, 126], [5, 94, 168, 126],
[109, 126], [5, 94, 126], [5, 126], [5, 81, 126], [5, 156, 176, 126],
[5, 111, 126], [5, 31, 126], [5, 6, 1, 126], [5, 89, 104, 176, 126],
[5, 176, 108, 126], [5, 72, 126], [5, 42, 92, 126], [5, 39, 126],
[5, 94, 126], [5, 126], [5, 1, 126], [5, 51, 126], [5, 101, 126],
[5, 111, 176, 126], [5, 94, 170, 176, 126], [5, 126], [5, 168, 126], [126],
[5, 183, 126], [5, 52, 168, 176, 126], [5, 94, 6, 126], [5, 94, 170, 126, 176],
[5, 126], [126], [5, 94, 4, 176], [5, 94, 176, 126], [5, 94, 185, 126], [5, 135, 126],
[5, 111, 163, 59, 126], [5, 107, 126], [5, 94, 159, 176, 126], [5, 94, 176, 126],
[5, 130, 176, 126], [5, 94, 176, 126], [5, 94, 176, 126], [5, 43, 176, 35, 126],
[5, 126], [5, 126], [5, 94, 126], [5, 183, 184, 176, 126], [5, 126], [5, 170, 22, 126],
[5, 94, 126], [5, 145, 126], [5, 126], [5, 180, 100, 126], [5, 183, 126],
[5, 94, 151, 126], [5, 44, 159, 126], [5, 6, 126], [5, 153, 126], [5, 94, 4, 126],
[5, 70, 126], [5, 31, 32, 6, 126], [5, 186, 187, 126], [5, 94, 126], [5, 97, 94, 176, 126],
[5, 94, 159, 176, 126], [5, 94, 126], [5, 94, 176, 126], [5, 37, 94, 111, 126], [5, 126],
[5, 94, 100, 126], [5, 94, 176, 126], [5, 126], [5, 126], [5, 94, 176, 126], [5, 62, 126],
[5, 94, 126], [5, 38, 126], [5, 126], [5, 172, 176, 126], [5, 168, 159, 52, 176, 126],
[5, 94, 13, 126], [5, 172, 126], [5, 94, 176, 126], [5, 126], [5, 3, 126], [5, 94, 176, 126],
[5, 36, 126], [5, 166, 126], [5, 16, 126], [5, 182, 37, 126], [5, 126], [5, 18, 126], [5, 126],
[5, 79, 34, 176, 126], [5, 94, 74, 176, 126], [5, 126], [5, 94, 176, 126], [5, 72, 39, 84, 176, 126],
[5, 143, 159, 176, 126], [5, 94, 159, 176, 126], [5, 118, 10, 176, 126], [5, 183, 176, 126],
[5, 1, 126], [5, 172, 126], [5, 94, 6, 126], [5, 94, 126], [5, 166, 155, 126], [5, 94, 136, 126],
[5, 52, 126], [5, 68, 176, 126], [5, 110, 126], [126], [5, 146, 39, 126], [5, 126], [5, 159, 176, 126],
[5, 183, 126], [5, 183, 126], [91, 126], [5, 31, 126], [5, 183, 126], [5, 137, 126], [5, 94, 126],
[5, 176, 95, 96, 126], [5, 176, 65, 126], [5, 79, 35, 176, 126], [5, 94, 170, 126], [5, 94, 176, 126],
[5, 94, 6, 126], [5, 113, 176, 126], [5, 41, 94, 126], [5, 94, 52, 126], [5, 94, 52, 126],
[5, 176, 55, 140, 126], [5, 94, 52, 77, 176, 126], [48, 126], [5, 63, 126], [150, 94, 23, 126],
[5, 94, 78, 37, 126], [5, 181, 126], [5, 57, 104, 126], [5, 6, 31, 176, 126], [5, 126], [5, 126],
[5, 72, 39, 126], [5, 94, 176, 126], [5, 94, 176, 126], [5, 129, 52, 126], [160, 126], [5, 172, 176, 126],
[5, 94, 172, 126], [5, 94, 126], [5, 94, 172, 126], [5, 126], [5, 176, 126], [5, 52, 126], [5, 172, 176, 126],
[5, 177, 20, 126], [5, 126], [5, 40, 176, 128, 126], [5, 94, 52, 37, 176, 126], [5, 81, 120, 150, 126],
[5, 170, 126], [5, 172, 126], [5, 61, 159, 126], [5, 159, 52, 126], [5, 111, 133, 176, 126],
[5, 94, 176, 170, 126], [5, 72, 126], [5, 111, 176, 126], [5, 172, 176, 126], [5, 183, 126],
[5, 94, 126], [5, 94, 176, 126], [5, 176, 126], [5, 126], [5, 113, 126], [168, 159, 52, 126],
[5, 138, 94, 159, 126], [7, 126], [5, 49, 94, 126], [5, 94, 159, 176, 126], [5, 126], [5, 111, 112, 176, 126],
[5, 170, 126], [5, 176, 126], [5, 94, 176, 126], [5, 175, 111, 126], [5, 126], [5, 72, 159, 58, 126],
[5, 138, 176, 126], [5, 94, 126], [5, 94, 52, 176, 126], [178, 182, 126], [5, 94, 126], [5, 45, 46, 126],
[5, 94, 176, 126], [5, 69, 159, 52, 126], [5, 168, 159, 176, 126], [5, 94, 39, 52, 176, 126], [5, 94, 65, 39, 126],
[5, 168, 52, 126], [5, 176, 126], [5, 28, 126], [5, 111, 112, 176, 126], [5, 113, 159, 52, 126],
[5, 163, 113, 59, 176, 126], [5, 176, 126], [5, 94, 159, 52, 126], [5, 94, 176, 126], [5, 111, 112, 126],
[5, 176, 170, 126], [5, 168, 159, 52, 176, 126], [5, 94, 147, 126], [5, 83, 176, 82, 126], [5, 94, 126],
[5, 59, 126], [5, 50, 52, 126], [5, 182, 163, 59, 126], [5, 126], [5, 126], [5, 159, 78, 126], [5, 1, 126],
[5, 159, 126], [5, 94, 78, 126], [5, 94, 4, 126], [5, 172, 126], [5, 172, 126], [5, 183, 0], [5, 168, 176, 126],
[5, 188, 189, 126], [5, 126], [5, 168, 126], [5, 52, 94, 176, 126], [5, 169, 126], [5, 31, 18, 176, 168, 126],
[5, 182, 126], [5, 124, 52, 126], [5, 182, 126], [5, 98, 126], [5, 94, 159, 126], [5, 6, 87, 126], [5, 94, 52, 126],
[5, 12, 126], [5, 94, 131, 176, 126], [5, 176, 126], [5, 126], [5, 126], [5, 94, 6, 126], [5, 182, 126],
[183, 184, 26, 126], [5, 126], [5, 138, 176, 126], [5, 172, 176, 126], [5, 126], [5, 126], [5, 34, 52, 176, 126],
[5, 148, 176, 126], [5, 105, 176, 126], [5, 94, 176, 126], [5, 94, 126], [5, 94, 168, 39, 126], [5, 94, 176, 126],
[5, 111, 126], [5, 26, 126], [5, 94, 52, 37, 126], [5, 183, 126], [5, 82, 126], [5, 182, 126], [5, 165, 52, 164, 126],
[5, 94, 126], [5, 3, 126], [5, 126], [5, 94, 172, 126], [5, 94, 126], [5, 76, 94, 126], [5, 47, 176, 126],
[5, 154, 176, 126], [5, 86, 126], [5, 126], [5, 94, 176, 126], [5, 172, 126], [5, 183, 126], [5, 71, 126],
[5, 94, 176, 126], [5, 126], [5, 94, 39, 126], [5, 126], [5, 6, 126], [5, 126, 92], [5, 172, 126], [5, 126],
[5, 163, 52, 126], [5, 176, 126], [5, 183, 126], [5, 52, 126], [5, 126], [5, 172, 126], [7, 126], [5, 94, 159, 52, 126],
[5, 94, 159, 52, 126], [5, 141, 176, 126], [5, 94, 126], [5, 114, 126], [5, 168, 170, 176, 126], [5, 94, 176, 126],
[5, 126], [5, 52, 94, 159, 126], [5, 94, 159, 52, 126], [5, 126], [5, 94, 138, 126], [5, 111, 163, 126],
[5, 2, 126], [5, 126], [5, 94, 126], [5, 92, 126], [5, 31, 126], [5, 6, 126], [5, 94, 126], [5, 60, 176, 126],
[5, 111, 116, 126], [5, 53, 176, 126], [5, 172, 176, 168, 126], [5, 94, 159, 176, 126], [5, 94, 66, 176, 126],
[5, 6, 126], [5, 94, 176, 126], [5, 176, 126], [5, 94, 149, 126], [5, 182, 176, 126], [5, 126], [5, 126],
[64, 94, 82, 176, 126], [5, 126], [5, 126], [5, 168, 52, 159, 126], [5, 79, 126], [5, 38, 126], [5, 144, 176, 126],
[5, 94, 176, 126], [17, 182, 126], [5, 150, 80, 126], [5, 94, 163, 59, 126], [5, 72, 126], [5, 111, 126],
[5, 94, 126], [5, 126, 94, 52], [5, 19, 126], [5, 1, 126], [5, 94, 78, 176, 126], [5, 30, 126], [5, 126],
[5, 52, 126], [5, 31, 126], [5, 52, 126], [5, 102, 39, 176, 126], [5, 94, 24, 126], [5, 101, 126], [5, 126],
[5, 111, 112, 6, 126], [5, 162, 126], [5, 158, 176, 4, 126], [5, 126], [5, 171, 126], [173, 166, 126],
[5, 11, 126], [5, 126], [5, 176, 125, 126], [5, 18, 126], [5, 94, 176, 126], [5, 94, 6, 126], [5, 126],
[5, 94, 176, 126], [5, 94, 39, 126], [5, 38, 126], [5, 168, 52, 176, 126], [5, 99, 126], [5, 117, 126, 176],
[5, 94, 39, 176, 126], [5, 90, 126], [5, 33, 126], [5, 168, 159, 52, 126], [5, 94, 176, 126], [5, 159, 113, 52, 126],
[5, 122, 176, 126], [5, 94, 176, 38, 126], [5, 126], [5, 31, 18, 126], [5, 9, 126], [5, 183, 184, 176, 126],
[5, 123, 126], [5, 72, 73, 84, 126], [5, 6, 126], [5, 155, 176, 126], [5, 111, 176, 126], [5, 94, 54, 176, 126],
[5, 106, 138, 176, 126], [5, 94, 176, 126], [5, 94, 126], [5, 32, 31, 126], [5, 132, 67, 172], [5, 103, 176, 126],
[5, 126], [5, 72, 39, 126], [5, 94, 126], [5, 183, 126], [5, 176, 126], [5, 126], [5, 94, 163, 59, 126], [5, 88, 126],
[5, 24, 126], [5, 142, 126], [75, 126], [5, 183, 134, 126], [5, 94, 126], [5, 183, 184, 126], [5, 94, 39, 126],
[24, 94, 176, 126], [5, 72, 39, 126], [5, 126], [5, 138, 139, 126]]

_C = CN()

_C.MODEL = CN()

###################
#### CoFormer  ####
###################

_C.MODEL.CoFormer = CN()
_C.MODEL.CoFormer.hidden_dim = 512
_C.MODEL.CoFormer.position_embedding = 'learned'
_C.MODEL.CoFormer.inference = True
_C.MODEL.CoFormer.backbone = 'resnet50'
_C.MODEL.CoFormer.dropout=0.15
_C.MODEL.CoFormer.nhead=8
_C.MODEL.CoFormer.num_glance_enc_layers=3
_C.MODEL.CoFormer.num_gaze_s1_dec_layers=3
_C.MODEL.CoFormer.num_gaze_s1_enc_layers=3
_C.MODEL.CoFormer.num_gaze_s2_dec_layers=3
_C.MODEL.CoFormer.dim_feedforward=2048
_C.MODEL.CoFormer.num_noun_classes = 9929
