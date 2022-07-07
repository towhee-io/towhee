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
import torch
from towhee.models.timesformer import TimeSformer
from towhee.models.timesformer.timesformer_utils import map_state_dict, get_configs


def create_model(
        model_name: str = 'svt_vitb_k400',
        pretrained: bool = False,
        checkpoint_path: str = None,
        device: str = None,
        **kwargs):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name is None:
        if pretrained:
            raise AssertionError('Fail to load pretrained model: no model name is specified.')
        model = TimeSformer(**kwargs)
    else:
        configs = get_configs(model_name)
        configs.update(**kwargs)
        model = TimeSformer(
            img_size=configs['img_size'],
            patch_size=configs['patch_size'],
            in_c=configs['in_c'],
            num_classes=configs['num_classes'],
            embed_dim=configs['embed_dim'],
            depth=configs['depth'],
            num_heads=configs['num_heads'],
            mlp_ratio=configs['mlp_ratio'],
            qkv_bias=configs['qkv_bias'],
            qk_scale=configs['qk_scale'],
            drop_ratio=configs['drop_ratio'],
            attn_drop_ratio=configs['attn_drop_ratio'],
            drop_path_ratio=configs['drop_path_ratio'],
            norm_layer=configs['norm_layer'],
            num_frames=configs['num_frames'],
            attention_type=configs['attention_type'],
            dropout=configs['dropout']
        )

    if pretrained:
        if checkpoint_path is None:
            raise AssertionError('Fail to load pretrained weights: no pretrained weights is specified.')
        state_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = map_state_dict(state_dict)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# if __name__ == '__main__':
#     path = '/Users/zilliz/PycharmProjects/pretrain/SVT/svt_k400_with_head.pth'
#     finetune_path = '/Users/zilliz/PycharmProjects/pretrain/SVT/finetune73/svt_k400_finetune.pth'
#     model = create_model(model_name='svt_vitb_k400', checkpoint_path=path,
#                          pretrained=True)
#     sample = torch.randn(1, 3, 8, 224, 224)
#     out1 = model(sample)
#     print(out1.shape)
