# modified from https://github.com/showlab/all-in-one/blob/main/AllInOne/modules/allinone_module.py
# Pytorch implementation of All in One: Exploring Unified Video-Language Pre-training

import torch
import torch.nn as nn
import math


class VCOPHeader(torch.nn.Module):
    def __init__(self, tuple_len=3, feature_size=768):
        """
        VCOPHeader for all-in-one
        """
        super(VCOPHeader, self).__init__()
        self.feature_size = feature_size
        self.fc7 = nn.Linear(self.feature_size * 2, 512)
        self.tuple_len = tuple_len
        pair_num = int(tuple_len * (tuple_len - 1) / 2)
        self.class_num = math.factorial(tuple_len)
        self.fc8 = nn.Linear(512 * pair_num, self.class_num)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        forward function
        """
        pf = []  # pairwise concat
        for i in range(self.tuple_len):
            for j in range(i + 1, self.tuple_len):
                pf.append(torch.cat([x[:, i], x[:, j]], dim=1))
        pf = [self.fc7(i) for i in pf]
        pf = [self.relu(i) for i in pf]
        h = torch.cat(pf, dim=1)
        h = self.dropout(h)
        h = self.fc8(h)  # logits
        return h
