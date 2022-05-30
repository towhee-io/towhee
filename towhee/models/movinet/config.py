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
# Inspired by https://github.com/PeizeSun/SparseR-CNN/blob/dff4c43a9526a6d0d2480abc833e78a7c29ddb1a/detectron2/config/defaults.py
#
# Original code from https://github.com/Atze00/MoViNet-pytorch
#
# Modified by Zilliz.

import os
try:
    from fvcore.common.config import CfgNode as CN
except ModuleNotFoundError:
    os.system("pip install fvcore")
    from fvcore.common.config import CfgNode as CN

def fill_se_config(conf, input_channels,
                    out_channels,
                    expanded_channels,
                    kernel_size,
                    stride,
                    padding,
                    padding_avg,
):
    conf.expanded_channels =expanded_channels
    conf.padding_avg= padding_avg
    fill_conv(conf,input_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
)

def fill_conv(conf, input_channels,
                out_channels,
                kernel_size,
                stride,
                padding,):
    conf.input_channels = input_channels
    conf.out_channels = out_channels
    conf.kernel_size = kernel_size
    conf.stride = stride
    conf.padding = padding

_C = CN()

_C.MODEL = CN()


###################
#### MoViNetA0 ####
###################

_C.MODEL.MoViNetA0 = CN()
_C.MODEL.MoViNetA0.name = "A0"
_C.MODEL.MoViNetA0.weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA0_statedict_v3?raw=true"
_C.MODEL.MoViNetA0.stream_weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA0_stream_statedict_v3?raw=true"
_C.MODEL.MoViNetA0.conv1 = CN()
fill_conv(_C.MODEL.MoViNetA0.conv1, 3,8,(1,3,3),(1,2,2),(0,1,1))


_C.MODEL.MoViNetA0.blocks = [ [CN()],
                            [CN() for _ in range(3)],
                            [CN() for _ in range(3)],
                            [CN() for _ in range(4)],
                            [CN() for _ in range(4)]]

#Block2
fill_se_config(_C.MODEL.MoViNetA0.blocks[0][0], 8, 8, 24, (1,5,5), (1,2,2), (0,2,2), (0,1,1))

#block 3
fill_se_config(_C.MODEL.MoViNetA0.blocks[1][0], 8, 32, 80, (3,3,3), (1,2,2), (1,0,0), (0,0,0))
fill_se_config(_C.MODEL.MoViNetA0.blocks[1][1], 32, 32, 80, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA0.blocks[1][2], 32, 32, 80, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 4
fill_se_config(_C.MODEL.MoViNetA0.blocks[2][0], 32, 56, 184, (5,3,3), (1,2,2), (2,0,0), (0,0,0))
fill_se_config(_C.MODEL.MoViNetA0.blocks[2][1], 56, 56, 112, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA0.blocks[2][2], 56, 56, 184, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 5
fill_se_config(_C.MODEL.MoViNetA0.blocks[3][0], 56, 56, 184, (5,3,3), (1,1,1), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA0.blocks[3][1], 56, 56, 184, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA0.blocks[3][2], 56, 56, 184, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA0.blocks[3][3], 56, 56, 184, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 6
fill_se_config(_C.MODEL.MoViNetA0.blocks[4][0], 56, 104, 384, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA0.blocks[4][1], 104, 104, 280, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA0.blocks[4][2], 104, 104, 280, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA0.blocks[4][3], 104, 104, 344, (1,5,5), (1,1,1), (0,2,2), (0,1,1))

_C.MODEL.MoViNetA0.conv7= CN()
fill_conv(_C.MODEL.MoViNetA0.conv7, 104,480,(1,1,1),(1,1,1),(0,0,0))

_C.MODEL.MoViNetA0.dense9= CN()
_C.MODEL.MoViNetA0.dense9.hidden_dim = 2048



###################
#### MoViNetA1 ####
###################

_C.MODEL.MoViNetA1 = CN()
_C.MODEL.MoViNetA1.name = "A1"
_C.MODEL.MoViNetA1.weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA1_statedict_v3?raw=true"
_C.MODEL.MoViNetA1.stream_weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA1_stream_statedict_v3?raw=true"
_C.MODEL.MoViNetA1.conv1 = CN()
fill_conv(_C.MODEL.MoViNetA1.conv1, 3, 16,(1,3,3),(1,2,2),(0,1,1))


_C.MODEL.MoViNetA1.blocks = [ [CN() for _ in range(2)],
                              [CN() for _ in range(4)],
                              [CN() for _ in range(5)],
                              [CN() for _ in range(6)],
                              [CN() for _ in range(7)]]


#Block2
fill_se_config(_C.MODEL.MoViNetA1.blocks[0][0], 16, 16, 40, (1,5,5), (1,2,2), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[0][1], 16, 16, 40, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 3
fill_se_config(_C.MODEL.MoViNetA1.blocks[1][0], 16, 40, 96, (3,3,3), (1,2,2), (1,0,0), (0,0,0))
fill_se_config(_C.MODEL.MoViNetA1.blocks[1][1], 40, 40, 120, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[1][2], 40, 40, 96, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[1][3], 40, 40, 96, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 4
fill_se_config(_C.MODEL.MoViNetA1.blocks[2][0], 40, 64, 216, (5,3,3), (1,2,2), (2,0,0), (0,0,0))
fill_se_config(_C.MODEL.MoViNetA1.blocks[2][1], 64, 64, 128, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[2][2], 64, 64, 216, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[2][3], 64, 64, 168, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[2][4], 64, 64, 216, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 5
fill_se_config(_C.MODEL.MoViNetA1.blocks[3][0], 64, 64, 216, (5,3,3), (1,1,1), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[3][1], 64, 64, 216, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[3][2], 64, 64, 216, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[3][3], 64, 64, 128, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[3][4], 64, 64, 128, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[3][5], 64, 64, 216, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 6
fill_se_config(_C.MODEL.MoViNetA1.blocks[4][0], 64 , 136, 456, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[4][1], 136, 136, 360, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[4][2], 136, 136, 360, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[4][3], 136, 136, 360, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[4][4], 136, 136, 456, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[4][5], 136, 136, 456, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA1.blocks[4][6], 136, 136, 544, (1,3,3), (1,1,1), (0,1,1), (0,1,1))

_C.MODEL.MoViNetA1.conv7= CN()
fill_conv(_C.MODEL.MoViNetA1.conv7, 136,600,(1,1,1),(1,1,1),(0,0,0))

_C.MODEL.MoViNetA1.dense9= CN()
_C.MODEL.MoViNetA1.dense9.hidden_dim = 2048



###################
#### MoViNetA2 ####
###################

_C.MODEL.MoViNetA2 = CN()
_C.MODEL.MoViNetA2.name = "A2"
_C.MODEL.MoViNetA2.weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA2_statedict_v3?raw=true"
_C.MODEL.MoViNetA2.stream_weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA2_stream_statedict_v3?raw=true"

_C.MODEL.MoViNetA2.conv1 = CN()
fill_conv(_C.MODEL.MoViNetA2.conv1, 3,16,(1,3,3),(1,2,2),(0,1,1))


_C.MODEL.MoViNetA2.blocks = [ [CN() for _ in range(3)],
                              [CN() for _ in range(5)],
                              [CN() for _ in range(5)],
                              [CN() for _ in range(6)],
                              [CN() for _ in range(7)]]

#Block2
fill_se_config(_C.MODEL.MoViNetA2.blocks[0][0], 16, 16, 40, (1,5,5), (1,2,2), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[0][1], 16, 16, 40, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[0][2], 16, 16, 64, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 3
fill_se_config(_C.MODEL.MoViNetA2.blocks[1][0], 16, 40, 96, (3,3,3), (1,2,2), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[1][1], 40, 40, 120, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[1][2], 40, 40, 96, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[1][3], 40, 40, 96, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[1][4], 40, 40, 120, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 4
fill_se_config(_C.MODEL.MoViNetA2.blocks[2][0], 40, 72, 240, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[2][1], 72, 72, 160, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[2][2], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[2][3], 72, 72, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[2][4], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 5
fill_se_config(_C.MODEL.MoViNetA2.blocks[3][0], 72, 72, 240, (5,3,3), (1,1,1), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[3][1], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[3][2], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[3][3], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[3][4], 72, 72, 144, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[3][5], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 6
fill_se_config(_C.MODEL.MoViNetA2.blocks[4][0], 72 , 144, 480, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[4][1], 144, 144, 384, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[4][2], 144, 144, 384, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[4][3], 144, 144, 480, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[4][4], 144, 144, 480, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[4][5], 144, 144, 480, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA2.blocks[4][6], 144, 144, 576, (1,3,3), (1,1,1), (0,1,1), (0,1,1))

_C.MODEL.MoViNetA2.conv7= CN()
fill_conv(_C.MODEL.MoViNetA2.conv7, 144,640,(1,1,1),(1,1,1),(0,0,0))

_C.MODEL.MoViNetA2.dense9= CN()
_C.MODEL.MoViNetA2.dense9.hidden_dim = 2048


###################
#### MoViNetA3 ####
###################

_C.MODEL.MoViNetA3 = CN()
_C.MODEL.MoViNetA3.name = "A3"
_C.MODEL.MoViNetA3.weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA3_statedict_v3?raw=true"
_C.MODEL.MoViNetA3.conv1 = CN()
fill_conv(_C.MODEL.MoViNetA3.conv1, 3,16,(1,3,3),(1,2,2),(0,1,1))


_C.MODEL.MoViNetA3.blocks =  [ [CN() for _ in range(4)],
                              [CN() for _ in range(6)],
                              [CN() for _ in range(5)],
                              [CN() for _ in range(8)],
                              [CN() for _ in range(10)]]

#Block2
fill_se_config(_C.MODEL.MoViNetA3.blocks[0][0], 16, 16, 40, (1,5,5), (1,2,2), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[0][1], 16, 16, 40, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[0][2], 16, 16, 64, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[0][3], 16, 16, 40, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 3
fill_se_config(_C.MODEL.MoViNetA3.blocks[1][0], 16, 48, 112, (3,3,3), (1,2,2), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[1][1], 48, 48, 144, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[1][2], 48, 48, 112, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[1][3], 48, 48, 112, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[1][4], 48, 48, 144, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[1][5], 48, 48, 144, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 4
fill_se_config(_C.MODEL.MoViNetA3.blocks[2][0], 48, 80, 240, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[2][1], 80, 80, 152, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[2][2], 80, 80, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[2][3], 80, 80, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[2][4], 80, 80, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 5
fill_se_config(_C.MODEL.MoViNetA3.blocks[3][0], 80, 88, 264, (5,3,3), (1,1,1), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[3][1], 88, 88, 264, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[3][2], 88, 88, 264, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[3][3], 88, 88, 264, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[3][4], 88, 88, 160, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[3][5], 88, 88, 264, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[3][6], 88, 88, 264, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[3][7], 88, 88, 264, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 6
fill_se_config(_C.MODEL.MoViNetA3.blocks[4][0], 88 , 168, 560, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[4][1], 168, 168, 448, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[4][2], 168, 168, 448, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[4][3], 168, 168, 560, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[4][4], 168, 168, 560, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[4][5], 168, 168, 560, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[4][6], 168, 168, 448, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[4][7], 168, 168, 448, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[4][8], 168, 168, 560, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA3.blocks[4][9], 168, 168, 672, (1,3,3), (1,1,1), (0,1,1), (0,1,1))

_C.MODEL.MoViNetA3.conv7= CN()
fill_conv(_C.MODEL.MoViNetA3.conv7, 168,744,(1,1,1),(1,1,1),(0,0,0))

_C.MODEL.MoViNetA3.dense9= CN()
_C.MODEL.MoViNetA3.dense9.hidden_dim = 2048

###################
#### MoViNetA4 ####
###################

_C.MODEL.MoViNetA4 = CN()
_C.MODEL.MoViNetA4.name = "A4"
_C.MODEL.MoViNetA4.weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA4_statedict_v3?raw=true"
_C.MODEL.MoViNetA4.conv1 = CN()
fill_conv(_C.MODEL.MoViNetA4.conv1, 3,24,(1,3,3),(1,2,2),(0,1,1))


_C.MODEL.MoViNetA4.blocks = [ [CN() for _ in range(6)],
                              [CN() for _ in range(9)],
                              [CN() for _ in range(9)],
                              [CN() for _ in range(10)],
                              [CN() for _ in range(13)]]

#Block2
fill_se_config(_C.MODEL.MoViNetA4.blocks[0][0], 24, 24, 64, (1,5,5), (1,2,2), (0,1,1), (0,0,0))
fill_se_config(_C.MODEL.MoViNetA4.blocks[0][1], 24, 24, 64, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[0][2], 24, 24, 96, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[0][3], 24, 24, 64, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[0][4], 24, 24, 96, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[0][5], 24, 24, 64, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 3
fill_se_config(_C.MODEL.MoViNetA4.blocks[1][0], 24, 56, 168, (3,3,3), (1,2,2), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[1][1], 56, 56, 168, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[1][2], 56, 56, 136, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[1][3], 56, 56, 136, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[1][4], 56, 56, 168, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[1][5], 56, 56, 168, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[1][6], 56, 56, 168, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[1][7], 56, 56, 136, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[1][8], 56, 56, 136, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 4
fill_se_config(_C.MODEL.MoViNetA4.blocks[2][0], 56, 96, 320, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[2][1], 96, 96, 160, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[2][2], 96, 96, 320, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[2][3], 96, 96, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[2][4], 96, 96, 320, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[2][5], 96, 96, 160, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[2][6], 96, 96, 320, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[2][7], 96, 96, 256, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[2][8], 96, 96, 320, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 5
fill_se_config(_C.MODEL.MoViNetA4.blocks[3][0], 96, 96, 320, (5,3,3), (1,1,1), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[3][1], 96, 96, 320, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[3][2], 96, 96, 320, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[3][3], 96, 96, 320, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[3][4], 96, 96, 192, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[3][5], 96, 96, 320, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[3][6], 96, 96, 320, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[3][7], 96, 96, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[3][8], 96, 96, 320, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[3][9], 96, 96, 320, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 6
fill_se_config(_C.MODEL.MoViNetA4.blocks[4][0], 96 , 192, 640, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[4][1], 192, 192, 512, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[4][2], 192, 192, 512, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[4][3], 192, 192, 640, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[4][4], 192, 192, 640, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[4][5], 192, 192, 640, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[4][6], 192, 192, 512, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[4][7], 192, 192, 512, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[4][8], 192, 192, 640, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[4][9], 192, 192, 768, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[4][10], 192, 192, 640, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[4][11], 192, 192, 640, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA4.blocks[4][12], 192, 192, 768, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

_C.MODEL.MoViNetA4.conv7= CN()
fill_conv(_C.MODEL.MoViNetA4.conv7, 192,856,(1,1,1),(1,1,1),(0,0,0))

_C.MODEL.MoViNetA4.dense9= CN()
_C.MODEL.MoViNetA4.dense9.hidden_dim = 2048


###################
#### MoViNetA5 ####
###################

_C.MODEL.MoViNetA5 = CN()
_C.MODEL.MoViNetA5.name = "A5"
_C.MODEL.MoViNetA5.weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA5_statedict_v3?raw=true"
_C.MODEL.MoViNetA5.conv1 = CN()
fill_conv(_C.MODEL.MoViNetA5.conv1, 3,24,(1,3,3),(1,2,2),(0,1,1))


_C.MODEL.MoViNetA5.blocks = [ [CN() for _ in range(6)],
                              [CN() for _ in range(11)],
                              [CN() for _ in range(13)],
                              [CN() for _ in range(11)],
                              [CN() for _ in range(18)]]

#Block2
fill_se_config(_C.MODEL.MoViNetA5.blocks[0][0], 24, 24, 64, (1,5,5), (1,2,2), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[0][1], 24, 24, 64, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[0][2], 24, 24, 96, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[0][3], 24, 24, 64, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[0][4], 24, 24, 96, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[0][5], 24, 24, 64, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 3
fill_se_config(_C.MODEL.MoViNetA5.blocks[1][0], 24, 64, 192, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[1][1], 64, 64, 152, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[1][2], 64, 64, 152, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[1][3], 64, 64, 152, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[1][4], 64, 64, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[1][5], 64, 64, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[1][6], 64, 64, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[1][7], 64, 64, 152, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[1][8], 64, 64, 152, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[1][9], 64, 64, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[1][10], 64, 64, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 4
fill_se_config(_C.MODEL.MoViNetA5.blocks[2][0], 64, 112, 376, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[2][1], 112, 112, 224, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[2][2], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[2][3], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[2][4], 112, 112, 296, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[2][5], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[2][6], 112, 112, 224, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[2][7], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[2][8], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[2][9], 112, 112, 296, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[2][10], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[2][11], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[2][12], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 5
fill_se_config(_C.MODEL.MoViNetA5.blocks[3][0], 112, 120, 376, (5,3,3), (1,1,1), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[3][1], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[3][2], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[3][3], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[3][4], 120, 120, 224, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[3][5], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[3][6], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[3][7], 120, 120, 224, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[3][8], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[3][9], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[3][10], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 6
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][0], 120 , 224, 744, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][1], 224, 224, 744, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][2], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][3], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][4], 224, 224, 744, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][5], 224, 224, 744, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][6], 224, 224, 744, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][7], 224, 224, 896, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][8], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][9], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][10], 224, 224, 896, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][11], 224, 224, 744, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][12], 224, 224, 744, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][13], 224, 224, 896, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][14], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][15], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][16], 224, 224, 744, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_se_config(_C.MODEL.MoViNetA5.blocks[4][17], 224, 224, 744, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

_C.MODEL.MoViNetA5.conv7= CN()
fill_conv(_C.MODEL.MoViNetA5.conv7, 224,992,(1,1,1),(1,1,1),(0,0,0))

_C.MODEL.MoViNetA5.dense9= CN()
_C.MODEL.MoViNetA5.dense9.hidden_dim = 2048
