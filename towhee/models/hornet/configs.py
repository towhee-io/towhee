# Configuration & model weights are from https://github.com/raoyongming/HorNet
#
# Copyright 2021 Zilliz. All rights reserved.
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

from functools import partial

from towhee.models.hornet import GatedConv, Block, GlobalLocalFilter


def get_configs(model_name):
    configs = {
        'in_chans': 3, 'num_classes': 1000, 'depths': (2, 3, 18, 2), 'base_dim': 128,
        'drop_path_rate': 0., 'layer_scale_init_value': 1e-6, 'head_init_scale': 1.,
        'gnconv': [
            partial(GatedConv, order=2, s=1.0 / 3.0),
            partial(GatedConv, order=3, s=1.0 / 3.0),
            partial(GatedConv, order=4, s=1.0 / 3.0),
            partial(GatedConv, order=5, s=1.0 / 3.0),
        ],
        'block': Block,
        'uniform_init': False
    }
    gf_gnconv = [
            partial(GatedConv, order=2, s=1.0 / 3.0),
            partial(GatedConv, order=3, s=1.0 / 3.0),
            partial(GatedConv, order=4, s=1.0 / 3.0, h=14, w=8, gflayer=GlobalLocalFilter),
            partial(GatedConv, order=5, s=1.0 / 3.0, h=7, w=4, gflayer=GlobalLocalFilter),
        ]

    if model_name is None:
        pass
    elif model_name == 'hornet_tiny_7x7':
        configs.update(
            url='https://cloud.tsinghua.edu.cn/f/1ca970586c6043709a3f/?dl=1',
            base_dim=64
        )
    elif model_name == 'hornet_tiny_gf':
        configs.update(
            url='https://cloud.tsinghua.edu.cn/f/511faad0bde94dfcaa54/?dl=1',
            base_dim=64,
            gnconv=gf_gnconv
        )
    elif model_name == 'hornet_small_7x7':
        configs.update(
            url='https://cloud.tsinghua.edu.cn/f/46422799db2941f7b684/?dl=1',
            base_dim=96
        )
    elif model_name == 'hornet_small_gf':
        configs.update(
            url='https://cloud.tsinghua.edu.cn/f/8405c984bf084d2ba85a/?dl=1',
            base_dim=96,
            gnconv=gf_gnconv
        )
    elif model_name == 'hornet_base_7x7':
        configs.update(
            url='https://cloud.tsinghua.edu.cn/f/5c86cb3d655d4c17a959/?dl=1'
        )
    elif model_name == 'hornet_base_gf':
        configs.update(
            url='https://cloud.tsinghua.edu.cn/f/6c84935e63b547f383fb/?dl=1',
            gnconv=gf_gnconv
        )
    elif model_name == 'hornet_base_gf_img384':
        configs.update(
            gnconv=[
                partial(GatedConv, order=2, s=1.0 / 3.0),
                partial(GatedConv, order=3, s=1.0 / 3.0),
                partial(GatedConv, order=4, s=1.0 / 3.0, h=24, w=13, gflayer=GlobalLocalFilter),
                partial(GatedConv, order=5, s=1.0 / 3.0, h=12, w=7, gflayer=GlobalLocalFilter),
            ]
        )
    elif model_name == 'hornet_large_7x7_22k':
        configs.update(
            url='https://cloud.tsinghua.edu.cn/f/4de41e26cb254c28a61a/?dl=1',
            num_classes=21841,
            base_dim=192
        )
    elif model_name == 'hornet_large_gf_22k':
        configs.update(
            url='https://cloud.tsinghua.edu.cn/f/8679b6acf63c41e285d9/?dl=1',
            num_classes=21841,
            base_dim=192,
            gnconv=gf_gnconv
        )
    elif model_name == 'hornet_large_gf_img384_22k':
        configs.update(
            url='https://cloud.tsinghua.edu.cn/f/f36957d46eef47da9c25/?dl=1',
            num_classes=21841,
            base_dim=192,
            gnconv=[
                partial(GatedConv, order=2, s=1.0 / 3.0),
                partial(GatedConv, order=3, s=1.0 / 3.0),
                partial(GatedConv, order=4, s=1.0 / 3.0, h=24, w=13, gflayer=GlobalLocalFilter),
                partial(GatedConv, order=5, s=1.0 / 3.0, h=12, w=7, gflayer=GlobalLocalFilter),
            ]
        )
    else:
        raise ValueError(f'Invalid model name: {model_name}.')
    return configs
