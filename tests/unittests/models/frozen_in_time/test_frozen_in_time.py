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

import unittest
import torch
from transformers import AutoTokenizer
from towhee.models.frozen_in_time import FrozenInTime
# if load frozen pretrained model , must import this package
# beacause this pretrained model is base on  https://github.com/victoresque/pytorch-template
from towhee.models.frozen_in_time.parse_config import ConfigParser
# this line for the pass pylint
ConfigParser()


class ForzenInTimeTest(unittest.TestCase):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # (batch ，frames，channels ， height ， width)
    dummy_video = torch.randn(1, 4, 3, 224, 224)
    text_pretrained_model_or_path = 'distilbert-base-uncased'
    text = ['开心每一天']
    tokenizer = AutoTokenizer.from_pretrained(text_pretrained_model_or_path, TOKENIZERS_PARALLELISM=False)
    dummy_text = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)

    def test_pretrained(self):
        '''
        use frozen in time pretrained model and Geotrend/distilbert-base-zh-cased pretrained model
        Returns:None
        '''

        model = FrozenInTime(attention_style='frozen_in_time',
                             is_pretrained=True,
                             weights_path='',
                             projection_dim=256,
                             text_pretrained_model=self.text_pretrained_model_or_path,
                             video_pretrained_model='vit_base_16x224',
                             video_model_type='SpaceTimeTransformer',
                             text_is_load_pretrained=True,
                             device=self.device)
        text_embeddings, video_embeddings = model(text=self.dummy_text, video=self.dummy_video, return_embeds=True)
        assert text_embeddings.shape == (1, 256)
        assert video_embeddings.shape == (1, 256)
        text_with_video_sim = model(text=self.dummy_text, video=self.dummy_video, return_embeds=False)
        print(text_with_video_sim)
        assert text_with_video_sim.shape == (1, 1)
        pass

    def test_without_pretrained(self):
        '''
        do not use frozen pretrained model
        Returns:None
        '''

        model = FrozenInTime(attention_style='frozen_in_time',
                             is_pretrained=False,
                             weights_path='',
                             projection_dim=256,
                             text_pretrained_model=self.text_pretrained_model_or_path,
                             video_pretrained_model='vit_base_16x224',
                             video_model_type='SpaceTimeTransformer',
                             text_is_load_pretrained=True,
                             device=self.device)
        text_embeddings, video_embeddings = model(text=self.dummy_text, video=self.dummy_video, return_embeds=True)
        assert text_embeddings.shape == (1, 256)
        assert video_embeddings.shape == (1, 256)
        text_with_video_sim = model(text=self.dummy_text, video=self.dummy_video, return_embeds=False)
        print(text_with_video_sim)
        assert text_with_video_sim.shape == (1, 1)


if __name__ == '__main__':
    unittest.main()
