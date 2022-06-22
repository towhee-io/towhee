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

import unittest
from collections import OrderedDict

import torch
from towhee.models.collaborative_experts import create_model


class TestCollaborativeExperts(unittest.TestCase):
    """
    Test CollaborativeExperts model
    """
    torch.manual_seed(1)
    text_dim = 3
    config = {
        "task": "retrieval",
        "use_ce": "pairwise",
        "text_dim": text_dim,
        "l2renorm": False,
        "expert_dims": OrderedDict([("audio", (1024, text_dim)),
                                    ("ocr", (12900, text_dim)), ("speech", (5700, text_dim))]),
        "vlad_clusters": {"ocr": 43, "audio": 8, "speech": 19, "text": 28},
        "ghost_clusters": {"ocr": 1, "audio": 1, "speech": 1, "text": 1},
        "disable_nan_checks": False,
        "keep_missing_modalities": False,
        "test_caption_mode": "indep",
        "randomise_feats": "",
        "feat_aggregation": {

            "ocr": {"model": "yang", "temporal": "vlad", "type": "embed", "flaky": True, "binarise": False,
                    "feat_dims": {"embed": 300}},
            "audio.vggish.0": {"model": "vggish", "flaky": True, "temporal": "vlad", "type": "embed",
                               "binarise": False},
            "audio": {"model": "vggish", "flaky": True, "temporal": "vlad", "type": "embed", "binarise": False},

            "speech": {"model": "w2v", "flaky": True, "temporal": "vlad", "type": "embed", "binarise": False,
                       "feat_dims": {"embed": 300}},
        },
        "ce_shared_dim": text_dim,
        "trn_config": {},
        "trn_cat": 0,
        "include_self": 1,
        "use_mish": 1,
        "use_bn_reason": 1,
        "num_h_layers": 0,
        "num_g_layers": 3,
        "kron_dets": False,
        "freeze_weights": False,
        "geometric_mlp": False,
        "rand_proj": False,
        "mimic_ce_dims": 0,
        "coord_dets": False,
        "concat_experts": False,
        "spatial_feats": False,
        "concat_mix_experts": False,
        "verbose": False,
        "num_classes": None,
    }

    ce_net = create_model(config=config)

    def test_forward(self):
        batch_size = 2
        experts = {"audio": torch.rand(batch_size, 29, 128),
                   "ocr": torch.rand(batch_size, 49, 300),
                   "speech": torch.rand(batch_size, 32, 300)
                   }
        ind = {
            "audio": torch.randint(low=0, high=2, size=(batch_size,)),
            "speech": torch.randint(low=0, high=2, size=(batch_size,)),
            "ocr": torch.randint(low=0, high=2, size=(batch_size,)),
        }
        text = torch.randn(batch_size, 1, 37, self.text_dim)
        out = self.ce_net(experts, ind, text)
        self.assertEqual(out["cross_view_conf_matrix"].shape, (batch_size, batch_size))

        self.assertEqual(out["text_embds"]["audio"].shape, (batch_size, 1, self.text_dim))

        self.assertEqual(out["vid_embds"]["audio"].shape, (batch_size, self.text_dim))


if __name__ == "__main__":
    unittest.main()
