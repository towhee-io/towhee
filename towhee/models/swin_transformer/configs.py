# Copyright 2021 Microsoft . All rights reserved.
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
# This code is modified by Zilliz.

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


model_cfgs = {
    # patch models (my experiments)
    'swin_base_patch4_window12_384': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),

    'swin_base_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth',
    ),

    'swin_large_patch4_window12_384': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),

    'swin_large_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth',
    ),

    'swin_small_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',
    ),

    'swin_tiny_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
    ),

    'swin_base_patch4_window12_384_in22k': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth',
        input_size=(3, 384, 384), crop_pct=1.0, num_classes=21841),

    'swin_base_patch4_window7_224_in22k': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
        num_classes=21841),

    'swin_large_patch4_window12_384_in22k': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
        input_size=(3, 384, 384), crop_pct=1.0, num_classes=21841),

    'swin_large_patch4_window7_224_in22k': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth',
        num_classes=21841),
    'swinv2_tiny_patch4_window8_256': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth',
        num_classes=21841),
    'swinv2_small_patch4_window8_256': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window8_256.pth',
        num_classes=21841),
    'swinv2_base_patch4_window8_256': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window8_256.pth',
        num_classes=21841),
    'swinv2_tiny_patch4_window16_256': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window16_256.pth',
        num_classes=21841),
    'swinv2_small_patch4_window16_256': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window16_256.pth',
        num_classes=21841),
    'swinv2_base_patch4_window16_256': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window16_256.pth',
        num_classes=21841),
    'swinv2_base_patch4_window12to16_192to256_22kto1k_ft': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth',
        num_classes=21841),
    'swinv2_base_patch4_window12to24_192to384_22kto1k_ft': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth',
        num_classes=21841),
    'swinv2_large_patch4_window12to16_192to256_22kto1k_ft': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12to16_192to256_22kto1k_ft.pth',
        num_classes=21841),
    'swinv2_large_patch4_window12to24_192to384_22kto1k_ft': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth',
        num_classes=21841),
    'swinv2_base_patch4_window12_192_22k': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12_192_22k.pth',
        num_classes=21841),
    'swinv2_large_patch4_window12_192_22k': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth',
        num_classes=21841),
}


def build_configs(name, **kwargs):

    config = model_cfgs[name]

    model_architectures = {
        'swin_base_patch4_window12_384': dict(
            patch_size=4, window_size=12, embed_dim=128, img_size=384, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
            **kwargs),

        'swin_base_patch4_window7_224' : dict(
            patch_size=4, window_size=7, embed_dim=128, img_size=224, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
            **kwargs),

        'swin_large_patch4_window12_384' : dict(
            patch_size=4, window_size=12, embed_dim=192, img_size=384, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
            **kwargs),

        'swin_large_patch4_window7_224' : dict(
            patch_size=4, window_size=7, embed_dim=192, img_size=224, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
            **kwargs),

        'swin_small_patch4_window7_224' : dict(
            patch_size=4, window_size=7, embed_dim=96, img_size=224, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24),
            **kwargs),

        'swin_tiny_patch4_window7_224' : dict(
            patch_size=4, window_size=7, embed_dim=96, img_size=224, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
            **kwargs),

        'swin_base_patch4_window12_384_in22k' : dict(
            patch_size=4, window_size=12, embed_dim=128, img_size=384, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
            **kwargs),

        'swin_base_patch4_window7_224_in22k' : dict(
            patch_size=4, window_size=7, embed_dim=128, img_size=224, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
            **kwargs),

        'swin_large_patch4_window12_384_in22k' : dict(
            patch_size=4, window_size=12, embed_dim=192, img_size=384, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
            **kwargs),

        'swin_large_patch4_window7_224_in22k' : dict(
            patch_size=4, window_size=7, embed_dim=192, img_size=224, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
            **kwargs),

        'swinv2_tiny_patch4_window8_256': dict(
            patch_size=4, window_size=8, embed_dim=96, img_size=256, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
            is_v2=True,
            **kwargs),
        'swinv2_small_patch4_window8_256': dict(
            patch_size=4, window_size=8, embed_dim=96, img_size=256, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24),
            is_v2=True,
            **kwargs),
        'swinv2_base_patch4_window8_256': dict(
            patch_size=4, window_size=8, embed_dim=128, img_size=256, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
            is_v2=True,
            **kwargs),
        'swinv2_tiny_patch4_window16_256': dict(
            patch_size=4, window_size=16, embed_dim=96, img_size=256, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
            is_v2=True,
            **kwargs),
        'swinv2_small_patch4_window16_256': dict(
            patch_size=4, window_size=16, embed_dim=96, img_size=256, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24),
            is_v2=True,
            **kwargs),
        'swinv2_base_patch4_window16_256': dict(
            patch_size=4, window_size=16, embed_dim=128, img_size=256, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
            is_v2=True,
            **kwargs),
        'swinv2_base_patch4_window12to16_192to256_22kto1k_ft': dict(
            patch_size=4, window_size=16, embed_dim=128, img_size=256, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
            is_v2=True, pretrained_window_sizes=[12, 12, 12, 6],
            **kwargs),
        'swinv2_base_patch4_window12to24_192to384_22kto1k_ft': dict(
            patch_size=4, window_size=24, embed_dim=128, img_size=384, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
            is_v2=True, pretrained_window_sizes=[12, 12, 12, 6],
            **kwargs),
        'swinv2_large_patch4_window12to16_192to256_22kto1k_ft': dict(
            patch_size=4, window_size=16, embed_dim=192, img_size=256, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
            is_v2=True, pretrained_window_sizes=[12, 12, 12, 6],
            **kwargs),
        'swinv2_large_patch4_window12to24_192to384_22kto1k_ft': dict(
            patch_size=4, window_size=24, embed_dim=192, img_size=384, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
            is_v2=True, pretrained_window_sizes=[12, 12, 12, 6],
            **kwargs),
        'swinv2_base_patch4_window12_192_22k': dict(
            patch_size=4, window_size=12, embed_dim=128, img_size=192, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
            is_v2=True,
            **kwargs),
        'swinv2_large_patch4_window12_192_22k': dict(
            patch_size=4, window_size=12, embed_dim=192, img_size=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
            is_v2=True,
            **kwargs),

    }
    return model_architectures[name], config
