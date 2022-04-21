# original code from https://github.com/krasserm/perceiver-io
# modified by Zilliz


from torch import nn


class Residual(nn.Module):
    """
    Residual module for Perceiver https://arxiv.org/pdf/2103.03206.pdf.

    Args:
        module (`nn.Module`):
            nn.Module.
        dropout (`nn.Module`):
            Dropout probability.
    """
    def __init__(self, module: nn.Module, dropout: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.dropout(x) + args[0]
