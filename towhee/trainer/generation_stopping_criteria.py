import time
import warnings
from abc import ABC
from copy import deepcopy
from typing import Optional

import torch

from .file_utils import add_start_docstrings


STOPPING_CRITERIA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
            or scores for each vocabulary token after SoftMax.
        kwargs:
            Additional stopping criteria specific kwargs.

    Return:
        :obj:`bool`. :obj:`False` indicates we should continue, :obj:`True` indicates we should stop.

"""


class StoppingCriteria(ABC):
    """Abstract base class for all stopping criteria that can be applied during generation."""

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


class MaxLengthCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds :obj:`max_length`.
    Keep in mind for decoder-only type of transformers, this will include the initial prompted tokens.

    Args:
        max_length (:obj:`int`):
            The maximum length that the output sequence can have in number of tokens.
    """

    def __init__(self, max_length: int):
        raise NotImplementedError

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        raise NotImplementedError


class MaxNewTokensCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the generated number of tokens exceeds :obj:`max_new_tokens`.
    Keep in mind for decoder-only type of transformers, this will **not** include the initial prompted tokens. This is
    very close to :obj:`MaxLengthCriteria` but ignores the number of initial tokens.

    Args:
        start_length (:obj:`int`):
            The number of initial tokens.
        max_new_tokens (:obj:`int`):
            The maximum number of tokens to generate.
    """

    def __init__(self, start_length: int, max_new_tokens: int):
        raise NotImplementedError

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        raise NotImplementedError


class MaxTimeCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generation exceeds some amount of time. By default, the
    time will start being counted when you initialize this function. You can override this by passing an
    :obj:`initial_time`.

    Args:
        max_time (:obj:`float`):
            The maximum allowed time in seconds for the generation.
        initial_time (:obj:`float`, `optional`, defaults to :obj:`time.time()`):
            The start of the generation allowed time.
    """

    def __init__(self, max_time: float, initial_timestamp: Optional[float] = None):
        raise NotImplementedError

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        raise NotImplementedError


class StoppingCriteriaList(list):
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        raise NotImplementedError

    @property
    def max_length(self) -> Optional[int]:
        raise NotImplementedError


def validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    raise NotImplementedError
