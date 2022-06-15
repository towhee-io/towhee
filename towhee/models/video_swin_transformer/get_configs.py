# Configs & Weights from https://github.com/SwinTransformer/Video-Swin-Transformer
#
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


def configs(model_name='swin_b_w877_k400_1k'):
    args = {
        'swin_b_k400_1k':
            {'pretrained': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_1k.pth',
             'num_classes': 400,
             'labels_file_name': 'kinetics_400.json',
             'embed_dim': 128,
             'depths': [2, 2, 18, 2],
             'num_heads': [4, 8, 16, 32],
             'patch_size': (2, 4, 4),
             'window_size': (8, 7, 7), 'drop_path_rate': 0.4, 'patch_norm': True},
        'swin_s_k400_1k':
            {
                'pretrained': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_small_patch244_window877_kinetics400_1k.pth',
                'num_classes': 400,
                'labels_file_name': 'kinetics_400.json',
                'embed_dim': 96,
                'depths': [2, 2, 18, 2],
                'num_heads': [3, 6, 12, 24],
                'patch_size': (2, 4, 4),
                'window_size': (8, 7, 7),
                'drop_path_rate': 0.4,
                'patch_norm': True},
        'swin_t_k400_1k':
            {
                'pretrained': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth',
                'num_classes': 400,
                'labels_file_name': 'kinetics_400.json',
                'embed_dim': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'patch_size': (2, 4, 4),
                'window_size': (8, 7, 7),
                'drop_path_rate': 0.1,
                'patch_norm': True},
        'swin_b_k400_22k':
            {
                'pretrained': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_22k.pth',
                'num_classes': 400,
                'labels_file_name': 'kinetics_400.json',
                'embed_dim': 128,
                'depths': [2, 2, 18, 2],
                'num_heads': [4, 8, 16, 32],
                'patch_size': (2, 4, 4),
                'window_size': (8, 7, 7),
                'drop_path_rate': 0.4,
                'patch_norm': True},
        'swin_b_k600_22k':
            {
                'pretrained': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth',
                'num_classes': 600,
                'labels_file_name': '',
                'embed_dim': 128,
                'depths': [2, 2, 18, 2],
                'num_heads': [4, 8, 16, 32],
                'patch_size': (2, 4, 4),
                'window_size': (8, 7, 7), 'drop_path_rate': 0.4, 'patch_norm': True},
        'swin_b_sthv2':
            {
                'pretrained': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_sthv2.pth',
                'num_classes': 174,
                'labels_file_name': '',
                'embed_dim': 128,
                'depths': [2, 2, 18, 2],
                'num_heads': [4, 8, 16, 32],
                'patch_size': (2, 4, 4),
                'window_size': (16, 7, 7),
                'drop_path_rate': 0.4,
                'patch_norm': True},
    }
    return args[model_name]

