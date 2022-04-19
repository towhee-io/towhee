# modified by Zilliz.

import torch

class Identity(torch.nn.Module):
    def forward(self, input_x):
        return input_x


class SegmentConsensus(torch.nn.Module):
    """
        SegmentConsensus class.
    """
    def __init__(self,
                 consensus_type,
                 dim=1):
        super().__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super().__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input_x):
        return SegmentConsensus(self.consensus_type, self.dim)(input_x)
