# original code from https://github.com/krasserm/perceiver-io


from torch import nn


class Sequential(nn.Sequential):
    """
    Sequential module for Perceiver https://arxiv.org/pdf/2103.03206.pdf.
    """

    def forward(self, *inputs):
        for module in self:
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
