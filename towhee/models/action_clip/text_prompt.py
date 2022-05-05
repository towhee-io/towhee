# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# Mengmeng Wang, Jiazheng Xing, Yong Liu
#
# Built on top of official implementation at https://github.com/sallymmx/ActionCLIP
#
# Modifications by Copyright 2021 Zilliz. All rights reserved.
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
from towhee.models import clip


def text_aug(word):
    augs = [f"a photo of action {word}", f"a picture of action {word}", f"Human action of {word}",
            f"{word}, an action", f"{word} this is an action", f"{word}, a video of action",
            f"Playing action of {word}", f"{word}", f"Playing a kind of action, {word}",
            f"Doing a kind of action, {word}", f"Look, the human is {word}",
            f"Can you recognize the action of {word}?", f"Video classification of {word}", f"A video of {word}",
            f"The man is {word}", f"The woman is {word}"]
    return augs


def text_prompt(labels):
    text_dict = {}
    num_txt_augs = len(text_aug(""))
    txt_augs = [text_aug(c) for c in labels]

    for i in range(num_txt_augs):
        vals = [clip.tokenize(augs[i]) for augs in txt_augs]
        text_dict[i] = torch.cat(vals)

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_txt_augs, text_dict
