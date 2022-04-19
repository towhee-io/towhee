# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# modified by Zilliz.

import torchvision
import numpy as np

import torch
from torch import nn
from torch.utils import model_zoo
from torch.nn.init import normal_, constant_

from towhee.models.tsm.basic_ops import ConsensusModule
from towhee.models.tsm.mobilenet_v2 import mobilenet_v2, InvertedResidual
from towhee.models.tsm.bn_inception import bninception
from towhee.models.tsm.temporal_shift import make_temporal_shift, TemporalShift
from towhee.models.tsm.non_local import make_non_local

class TSN(nn.Module):
    """
    Args:
        num_class ('int'):
            Output class nums.
        num_segments ('int'):
             The temperal shift segents.
        modality ('string'):
            Modality of the video like "RGB".
        base_model ('string'):
            The base model of TSM model. Default: 'resnet101'
        new_length ('bool')
            If true, RGBDiff needs one more image to calculate diff. Default: None
        consensus_type ('string'):
            The type of consensus. Default: 'avg'
        before_softmax ('bool')
            If true, shift operation works before softmax. Default: True
        dropout ('float')
            Dropout rate. Default: 0.8,
        img_feature_dim ('int'):
            The feature dimension size of input images. Default: 256
        crop_num ('int'):
            The crop numbers of video. Default: 1
        partial_bn ('bool')
            If True, add partial batch normalization. Default: True
        pretrain ('string'):
            pretrained dataset name. Default: 'imagenet'
        is_shift ('bool')
            If True, add temperal shift to the model. Default: False
        shift_div ('int'):
            Scale reciprocal of shift operation. Default: 8
        shift_place ('string'):
            Shift place. Default: 'blockres'
        fc_lr5 ('bool')
            If True, the parameters of the last fc layer in cls_head have 5x lr multiplier
            and 10x weight decay multiplier. Default: False.
        temporal_pool ('bool')
            If true ,use temperal_pool. Default: False
        non_local ('bool')
            If true, use the non_local block. Default: False
    """
    def __init__(self,
                 num_class,
                 num_segments,
                 modality,
                 base_model='resnet101',
                 new_length=None,
                 consensus_type='avg',
                 before_softmax=True,
                 dropout=0.8,
                 img_feature_dim=256,
                 crop_num=1,
                 partial_bn=True,
                 pretrain='imagenet',
                 is_shift=True,
                 shift_div=8,
                 shift_place='blockres',
                 fc_lr5=False,
                 temporal_pool=False,
                 non_local=True):
        super().__init__()

        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        if not before_softmax and consensus_type != 'avg':
            raise ValueError('Only avg consensus can be used after Softmax')

        if new_length is None:
            self.new_length = 1 if modality == 'RGB' else 5
        else:
            self.new_length = new_length

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print('Converting the ImageNet model to a flow init model')
            self.base_model = self._construct_flow_model(self.base_model)
            print('Done. Flow model ready...')
        elif self.modality == 'RGBDiff':
            print('Converting the ImageNet model to RGB+Diff init model')
            self.base_model = self._construct_diff_model(self.base_model)
            print('Done. RGBDiff model ready.')

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialbn(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model,
                    self.base_model.last_layer_name,
                    nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)
            if self.is_shift:
                #print('Adding temporal shift...')
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div,
                                    place=self.shift_place,
                                    temporal_pool=self.temporal_pool)

            if self.non_local:
                #print('Adding non-local module...')
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std +\
                    [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'mobilenetv2':
            self.base_model = mobilenet_v2(bool(self.pretrain == 'imagenet'))

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.is_shift:
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        if self.print_spec:
                            print('Adding temporal shift... {}'.format(m.use_res_connect))
                        m.conv[0] = TemporalShift(m.conv[0],
                                                  n_segment=self.num_segments,
                                                  n_div=self.shift_div)
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + \
                    [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'BNInception':
            self.base_model = bninception(pretrained=self.pretrain)
            self.input_size = self.base_model.input_size
            self.input_mean = self.base_model.mean
            self.input_std = self.base_model.std
            self.base_model.last_layer_name = 'fc'
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
            if self.is_shift:
                print('Adding temporal shift...')
                self.base_model.build_temporal_ops(
                    self.num_segments, is_temporal_shift=self.shift_place, shift_div=self.shift_div)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def partialbn(self, enable):
        self._enable_pbn = enable

    def forward(self, input_x , no_reshape=False):
        if not no_reshape:
            sample_len = (3 if self.modality == 'RGB' else 2) * self.new_length

            if self.modality == 'RGBDiff':
                sample_len = 3 * self.new_length
                input_x = self._get_diff(input_x)

            base_out = self.base_model(input_x.view((-1, sample_len) + input_x.size()[-2:]))
        else:
            base_out = self.base_model(input_x)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
            if self.is_shift and self.temporal_pool:
                base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
            else:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            output = self.consensus(base_out)
            return output.squeeze(1)

    def _get_diff(self, input_x, keep_rgb=False):
        input_c = 3 if self.modality in ['RGB', 'RGBDiff'] else 2
        input_view = input_x.view((-1,
                                 self.num_segments,
                                 self.new_length + 1,
                                 input_c,)\
                                 + input_x.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :]\
                    - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :]\
                    - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x],
                              nn.Conv2d),
                              list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=bool(len(params) == 2))
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            # add bias if neccessary
            new_conv.bias.data = params[1].data

        # remove .weight suffix to get the layer name
        layer_name = list(container.state_dict().keys())[0][:-7]

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_model_name == 'BNInception':
            model_dir = model_zoo.load_url(
            'https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1')
            base_model.load_state_dict(model_dir)
            print('=> Loading pretrained Flow weight done...')
        else:
            print('#' * 30, 'Warning! No Flow pretrained model is found')
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x],
                                nn.Conv2d),
                                list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1]\
            + (3 * self.new_length,)\
            + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True)\
                .expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1]\
            + (3 * self.new_length,)\
            + kernel_size[2:]
            new_kernels = torch.cat((params[0].data,
                                     params[0].data.mean(dim=1, keepdim=True)\
                                         .expand(new_kernel_size).contiguous()),
                                     1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bool(len(params) == 2))
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            # add bias if neccessary
            new_conv.bias.data = params[1].data
        # remove .weight suffix to get the layer name
        layer_name = list(container.state_dict().keys())[0][:-7]

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

if __name__=='__main__':
    model=TSN(10,3,'RGB')
    print(model)
