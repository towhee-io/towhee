# Original pytorch implementation by:
# 'Correlation Verification for Image Retrieval'
#       - https://arxiv.org/abs/2204.01458
# Original code by / Copyright 2022, Seongwon Lee.
# Modifications & additions by / Copyright 2022 Zilliz. All rights reserved.
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

from functools import reduce
from operator import add
import torch
from torch import nn

from towhee.models.cvnet.cvnet_utils import extract_feat_res_pycls, get_configs
from towhee.models.cvnet.cvnet_block import CVLearner, Correlation
from towhee.models.cvnet.resnet import ResNet
from towhee.models.utils import create_model as towhee_model


class CVNet(nn.Module):
    """
    CVNet

    Args:
        resnet_depth (`int`):
            ResNet depth.
        reduction_dim (`int`):
            Reduction dimension of ResNet.
    """
    def __init__(self, resnet_depth=50, reduction_dim=2048):
        super().__init__()

        self.encoder_q = ResNet(resnet_depth, reduction_dim)
        self.encoder_q.eval()

        self.scales = [0.25, 0.5, 1.0]
        self.num_scales = len(self.scales)

        feat_dim_l3 = 1024
        self.channel_compressed = 256

        self.softmax = nn.Softmax(dim=1)
        self.extract_feats = extract_feat_res_pycls

        if resnet_depth == 50:
            nbottlenecks = [3, 4, 6, 3]
            self.feat_ids = [13]
        elif resnet_depth == 101:
            nbottlenecks = [3, 4, 23, 3]
            self.feat_ids = [30]
        else:
            raise Exception("Unavailable RESNET_DEPTH %s" % resnet_depth)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])

        self.conv2ds = nn.ModuleList([nn.Conv2d(feat_dim_l3, 256, kernel_size=3, padding=1, bias=False) for _ in self.scales])

        self.cv_learner = CVLearner([self.num_scales*self.num_scales, self.num_scales*self.num_scales, self.num_scales*self.num_scales])

    def forward(self, query_img, key_img):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.encoder_q, self.feat_ids, self.bottleneck_ids, self.lids)
            key_feats = self.extract_feats(key_img, self.encoder_q, self.feat_ids, self.bottleneck_ids, self.lids)
            corr_qk = Correlation.build_crossscale_correlation(query_feats[0], key_feats[0], self.scales, self.conv2ds)
            logits_qk = self.cv_learner(corr_qk)
            score = self.softmax(logits_qk)[:,1]
        return score

    def extract_global_descriptor(self, im_q):
        # compute query features
        q = self.encoder_q(im_q)[0]
        q = nn.functional.normalize(q, dim=1)
        return q

    def extract_featuremap(self, img):
        with torch.no_grad():
            feats = self.extract_feats(img, self.encoder_q, self.feat_ids, self.bottleneck_ids, self.lids)
        return feats

    def extract_score_with_featuremap(self, query_feats, key_feats):
        with torch.no_grad():
            corr_qk = Correlation.build_crossscale_correlation(query_feats[0], key_feats[0], self.scales, self.conv2ds)
            logits_qk = self.cv_learner(corr_qk)
            score = self.softmax(logits_qk)[0][1]
        return score


def create_model(
        model_name: str = None,
        pretrained: bool = False,
        checkpoint_path: str = None,
        device: str = None,
        **kwargs
        ):
    configs = get_configs(model_name)
    configs.update(**kwargs)

    model = towhee_model(CVNet, configs=configs, pretrained=pretrained, checkpoint_path=checkpoint_path, device=device)
    return model


# if __name__ == '__main__':
#     path1 = '/Users/zilliz/PycharmProjects/pretrain/CVNet/CVPR2022_CVNet_R101.pyth'
#     path2 = '/Users/zilliz/PycharmProjects/pretrain/CVNet/CVPR2022_CVNet_R50.pyth'
#     model = create_model(model_name='CVNet_R101', pretrained=True, weights_path=path1)
#     query_image = torch.randn(1, 3, 224, 224)
#     key_image = torch.randn(1, 3, 224, 224)
#     score = model(query_image, key_image)
#     score = score.unsqueeze(-1)
#     print(score)
#     print(score.shape)

