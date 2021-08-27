# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team
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

import inspect
import math
from abc import ABC
from typing import Callable, Iterable, List

import numpy as np
import torch

from .file_utils import add_start_docstrings
from .logging import get_logger


logger = get_logger(__name__)


LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search
        kwargs:
            Additional logits processor specific kwargs.

    Return:
        :obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.vocab_size)`: The processed prediction scores.

"""


class LogitsProcessor(ABC):
    """Abstract base class for all logit processors that can be applied during generation."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Torch method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsWarper(ABC):
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Torch method for warping logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list):
    """
    This class can be used to create a list of :class:`~transformers.LogitsProcessor` or
    :class:`~towhee.LogitsWarper` to subsequently process a :obj:`scores` input tensor. This class inherits from
    list and adds a specific `__call__` method to apply each :class:`~transformers.LogitsProcessor` or
    :class:`~transformers.LogitsWarper` to the inputs.
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        raise NotImplementedError


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    :class:`towhee.LogitsProcessor` enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (:obj:`int`):
            The minimum length below which the score of :obj:`eos_token_id` is set to :obj:`-float("Inf")`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, min_length: int, eos_token_id: int):
        raise NotImplementedError

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


class TemperatureLogitsWarper(LogitsWarper):
    r"""
    :class:`towhee.LogitsWarper` for temperature (exponential scaling output probability distribution).

    Args:
        temperature (:obj:`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        raise NotImplementedError

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        raise NotImplementedError


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    :class:`towhee.LogitsProcessor` enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (:obj:`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
    """

    def __init__(self, penalty: float):
        raise NotImplementedError

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


class TopPLogitsWarper(LogitsWarper):
    """
    :class:`towhee.LogitsWarper` that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <=
    prob_cut_off.

    Args:
        top_p (:obj:`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or higher are
            kept for generation.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        raise NotImplementedError

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


class TopKLogitsWarper(LogitsWarper):
    r"""
    :class:`towhee.LogitsWarper` that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (:obj:`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        raise NotImplementedError

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    raise NotImplementedError


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    raise NotImplementedError


def _calc_banned_ngram_tokens(
    ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int
) -> List[Iterable[int]]:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    raise NotImplementedError


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` that enforces no repetition of n-grams. See `Fairseq
    <https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345>`__.

    Args:
        ngram_size (:obj:`int`):
            All ngrams of size :obj:`ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        raise NotImplementedError

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


class EncoderNoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` that enforces no repetition of encoder input ids n-grams for the decoder ids.
    See `ParlAI <https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/torch_generator_agent.py#L1350>`__.

    Args:
        encoder_ngram_size (:obj:`int`):
            All ngrams of size :obj:`ngram_size` can only occur within the encoder input ids.
        encoder_input_ids (:obj:`int`):
            The encoder_input_ids that should not be repeated within the decoder ids.
    """

    def __init__(self, encoder_ngram_size: int, encoder_input_ids: torch.LongTensor):
        raise NotImplementedError

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


class NoBadWordsLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsProcessor` that enforces that specified sequences will never be sampled.

    Args:
        bad_words_ids (:obj:`List[List[int]]`):
            List of list of token ids that are not allowed to be generated. In order to get the tokens of the words
            that should not appear in the generated text, use :obj:`tokenizer(bad_word,
            add_prefix_space=True).input_ids`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, bad_words_ids: List[List[int]], eos_token_id: int):
        raise NotImplementedError

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def _tokens_match(self, prev_tokens: torch.LongTensor, tokens: List[int]) -> bool:
        raise NotImplementedError

    def _calc_banned_bad_words_ids(self, prev_input_ids: Iterable[int]) -> Iterable[int]:
        raise NotImplementedError

    def _set_scores_to_inf_for_banned_tokens(self, scores: torch.Tensor, banned_tokens: List[List[int]]) -> None:
        """
        Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be a
        list of list of banned tokens to ban in the format [[batch index, vocabulary position],...

        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
        """
        raise NotImplementedError


class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` that enforces constrained generation and is useful for prefix-conditioned
    constrained generation. See `Autoregressive Entity Retrieval <https://arxiv.org/abs/2010.00904>`__ for more
    information.

    Args:
        prefix_allowed_tokens_fn: (:obj:`Callable[[int, torch.Tensor], List[int]]`):
            This function constraints the beam search to allowed tokens only at each step. This function takes 2
            arguments :obj:`inputs_ids` and the batch ID :obj:`batch_id`. It has to return a list with the allowed
            tokens for the next generation step conditioned on the previously generated tokens :obj:`inputs_ids` and
            the batch ID :obj:`batch_id`.
    """

    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int):
        raise NotImplementedError

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


class HammingDiversityLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` that enforces diverse beam search. Note that this logits processor is only
    effective for :meth:`transformers.PreTrainedModel.group_beam_search`. See `Diverse Beam Search: Decoding Diverse
    Solutions from Neural Sequence Models <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.

    Args:
        diversity_penalty (:obj:`float`):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group at a
            particular time. Note that :obj:`diversity_penalty` is only effective if ``group beam search`` is enabled.
        num_beams (:obj:`int`):
            Number of beams used for group beam search. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for
            more details.
        num_beam_groups (:obj:`int`):
            Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
            beams. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
    """

    def __init__(self, diversity_penalty: float, num_beams: int, num_beam_groups: int):
        raise NotImplementedError

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        current_tokens: torch.LongTensor,
        beam_group_idx: int,
    ) -> torch.FloatTensor:
        raise NotImplementedError


class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    r"""
    :class:`~transformers.LogitsProcessor` that enforces the specified token as the first generated token.

    Args:
        bos_token_id (:obj:`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, bos_token_id: int):
        raise NotImplementedError

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    r"""
    :class:`~transformers.LogitsProcessor` that enforces the specified token as the last generated token when
    :obj:`max_length` is reached.

    Args:
        max_length (:obj:`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (:obj:`int`):
            The id of the token to force as the last generated token when :obj:`max_length` is reached.
    """

    def __init__(self, max_length: int, eos_token_id: int):
        raise NotImplementedError

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


class InfNanRemoveLogitsProcessor(LogitsProcessor):
    r"""
    :class:`~transformers.LogitsProcessor` that removes all :obj:`nan` and :obj:`inf` values to avoid the generation
    method to fail. Note that using the logits processor should only be used if necessary since it can slow down the
    generation method. :obj:`max_length` is reached.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError
