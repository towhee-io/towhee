# coding=utf-8
# Copyright 2021 Zilliz and The HuggingFace Inc. team.
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
"""
Base classes common to both the slow and the fast tokenization classes: PreTrainedTokenizerBase (host all the user
fronting encoding methods) Special token mixing (host the special tokens logic) and BatchEncoding (wrap the dictionary
of output with special method for the Fast tokenizers)
"""

import copy
import json
import os
import re
import warnings
from collections import OrderedDict, UserDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from packaging import version

import requests

from . import __version__
from .file_utils import (
    ExplicitEnum,
    PaddingStrategy,
    PushToHubMixin,
    TensorType,
    _is_jax,
    _is_numpy,
    _is_tensorflow,
    _is_torch,
    _is_torch_device,
    add_end_docstrings,
    cached_path,
    copy_func,
    get_list_of_files,
    hf_bucket_url,
    is_flax_available,
    is_offline_mode,
    is_remote_url,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    to_py_obj,
    torch_required,
)
import logging


if TYPE_CHECKING:
    if is_torch_available():
        import torch
    if is_tf_available():
        import tensorflow as tf
    if is_flax_available():
        import jax.numpy as jnp  # noqa: F401


if is_tokenizers_available():
    from tokenizers import AddedToken
    from tokenizers import Encoding as EncodingFast
else:

    @dataclass(frozen=True, eq=True)
    class AddedToken:
        """
        AddedToken represents a token to be added to a Tokenizer An AddedToken can have special options defining the
        way it should behave.
        """

        content: str = field(default_factory=str)
        single_word: bool = False
        lstrip: bool = False
        rstrip: bool = False
        normalized: bool = True

        def __getstate__(self):
            return self.__dict__

    @dataclass
    class EncodingFast:
        """This is dummy class because without the `tokenizers` library we don't have these objects anyway"""

        pass


logger = logging.get_logger(__name__)

VERY_LARGE_INTEGER = int(1e30)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(1e20)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]


# Slow tokenizers used to be saved in three separated files
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

# Fast tokenizers (provided by HuggingFace tokenizer's library) can be saved in a single file
FULL_TOKENIZER_FILE = "tokenizer.json"
_re_tokenizer_file = re.compile(r"tokenizer\.(.*)\.json")


class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the ``truncation`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for
    tab-completion in an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


class CharSpan(NamedTuple):
    """
    Character span in the original string.

    Args:
        start (:obj:`int`): Index of the first character in the original string.
        end (:obj:`int`): Index of the character following the last character in the original string.
    """

    start: int
    end: int


class TokenSpan(NamedTuple):
    """
    Token span in an encoded string (list of tokens).

    Args:
        start (:obj:`int`): Index of the first token in the span.
        end (:obj:`int`): Index of the token following the last token in the span.
    """

    start: int
    end: int


class BatchEncoding(UserDict):
    """
    Holds the output of the :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase.encode_plus` and
    :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase.batch_encode` methods (tokens,
    attention_masks, etc).

    This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes
    utility methods to map from word/character space to token space.

    Args:
        data (:obj:`dict`):
            Dictionary of lists/arrays/tensors returned by the encode/batch_encode methods ('input_ids',
            'attention_mask', etc.).
        encoding (:obj:`tokenizers.Encoding` or :obj:`Sequence[tokenizers.Encoding]`, `optional`):
            If the tokenizer is a fast tokenizer which outputs additional information like mapping from word/character
            space to token space the :obj:`tokenizers.Encoding` instance or list of instance (for batches) hold this
            information.
        tensor_type (:obj:`Union[None, str, TensorType]`, `optional`):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
        prepend_batch_axis (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to add a batch axis when converting to tensors (see :obj:`tensor_type` above).
        n_sequences (:obj:`Optional[int]`, `optional`):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        encoding: Optional[Union[EncodingFast, Sequence[EncodingFast]]] = None,
        tensor_type: Union[None, str, TensorType] = None,
        prepend_batch_axis: bool = False,
        n_sequences: Optional[int] = None,
    ):
        raise NotImplementedError

    @property
    def n_sequences(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: The number of sequences used to generate each sample from the batch encoded in this
        :class:`~transformers.BatchEncoding`. Currently can be one of :obj:`None` (unknown), :obj:`1` (a single
        sentence) or :obj:`2` (a pair of sentences)
        """
        raise NotImplementedError

    @property
    def is_fast(self) -> bool:
        """
        :obj:`bool`: Indicate whether this :class:`~transformers.BatchEncoding` was generated from the result of a
        :class:`~transformers.PreTrainedTokenizerFast` or not.
        """
        raise NotImplementedError

    def __getitem__(self, item: Union[int, str]) -> Union[Any, EncodingFast]:
        """
        If the key is a string, returns the value of the dict associated to :obj:`key` ('input_ids', 'attention_mask',
        etc.).

        If the key is an integer, get the :obj:`tokenizers.Encoding` for batch item with index :obj:`key`.
        """
        raise NotImplementedError

    def __getattr__(self, item: str):
        raise NotImplementedError

    def __getstate__(self):
        raise NotImplementedError

    def __setstate__(self, state):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def values(self):
        raise NotImplementedError

    def items(self):
        raise NotImplementedError

    # After this point:
    # Extended properties and methods only available for fast (Rust-based) tokenizers
    # provided by HuggingFace tokenizers library.

    @property
    def encodings(self) -> Optional[List[EncodingFast]]:
        """
        :obj:`Optional[List[tokenizers.Encoding]]`: The list all encodings from the tokenization process. Returns
        :obj:`None` if the input was tokenized through Python (i.e., not a fast) tokenizer.
        """
        raise NotImplementedError

    def tokens(self, batch_index: int = 0) -> List[str]:
        """
        Return the list of tokens (sub-parts of the input strings after word/subword splitting and before conversion to
        integer indices) at a given batch index (only works for the output of a fast tokenizer).

        Args:
            batch_index (:obj:`int`, `optional`, defaults to 0): The index to access in the batch.

        Returns:
            :obj:`List[str]`: The list of tokens at that index.
        """
        raise NotImplementedError

    def sequence_ids(self, batch_index: int = 0) -> List[Optional[int]]:
        """
        Return a list mapping the tokens to the id of their original sentences:

            - :obj:`None` for special tokens added around or between sequences,
            - :obj:`0` for tokens corresponding to words in the first sequence,
            - :obj:`1` for tokens corresponding to words in the second sequence when a pair of sequences was jointly
              encoded.

        Args:
            batch_index (:obj:`int`, `optional`, defaults to 0): The index to access in the batch.

        Returns:
            :obj:`List[Optional[int]]`: A list indicating the sequence id corresponding to each token. Special tokens
            added by the tokenizer are mapped to :obj:`None` and other tokens are mapped to the index of their
            corresponding sequence.
        """
        raise NotImplementedError

    def words(self, batch_index: int = 0) -> List[Optional[int]]:
        """
        Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.

        Args:
            batch_index (:obj:`int`, `optional`, defaults to 0): The index to access in the batch.

        Returns:
            :obj:`List[Optional[int]]`: A list indicating the word corresponding to each token. Special tokens added by
            the tokenizer are mapped to :obj:`None` and other tokens are mapped to the index of their corresponding
            word (several tokens will be mapped to the same word index if they are parts of that word).
        """
        raise NotImplementedError

    def word_ids(self, batch_index: int = 0) -> List[Optional[int]]:
        """
        Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.

        Args:
            batch_index (:obj:`int`, `optional`, defaults to 0): The index to access in the batch.

        Returns:
            :obj:`List[Optional[int]]`: A list indicating the word corresponding to each token. Special tokens added by
            the tokenizer are mapped to :obj:`None` and other tokens are mapped to the index of their corresponding
            word (several tokens will be mapped to the same word index if they are parts of that word).
        """
        raise NotImplementedError

    def token_to_sequence(self, batch_or_token_index: int, token_index: Optional[int] = None) -> int:
        """
        Get the index of the sequence represented by the given token. In the general use case, this method returns
        :obj:`0` for a single sequence or the first sequence of a pair, and :obj:`1` for the second sequence of a pair

        Can be called as:

        - ``self.token_to_sequence(token_index)`` if batch size is 1
        - ``self.token_to_sequence(batch_index, token_index)`` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
        words are defined by the user). In this case it allows to easily associate encoded tokens with provided
        tokenized words.

        Args:
            batch_or_token_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
                the token in the sequence.
            token_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index of the token in the
                sequence.

        Returns:
            :obj:`int`: Index of the word in the input sequence.
        """

        raise NotImplementedError

    def token_to_word(self, batch_or_token_index: int, token_index: Optional[int] = None) -> int:
        """
        Get the index of the word corresponding (i.e. comprising) to an encoded token in a sequence of the batch.

        Can be called as:

        - ``self.token_to_word(token_index)`` if batch size is 1
        - ``self.token_to_word(batch_index, token_index)`` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
        words are defined by the user). In this case it allows to easily associate encoded tokens with provided
        tokenized words.

        Args:
            batch_or_token_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the token in the sequence.
            token_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index of the token in the
                sequence.

        Returns:
            :obj:`int`: Index of the word in the input sequence.
        """

        raise NotImplementedError

    def word_to_tokens(
        self, batch_or_word_index: int, word_index: Optional[int] = None, sequence_index: int = 0
    ) -> Optional[TokenSpan]:
        """
        Get the encoded token span corresponding to a word in a sequence of the batch.

        Token spans are returned as a :class:`~transformers.tokenization_utils_base.TokenSpan` with:

        - **start** -- Index of the first token.
        - **end** -- Index of the token following the last token.

        Can be called as:

        - ``self.word_to_tokens(word_index, sequence_index: int = 0)`` if batch size is 1
        - ``self.word_to_tokens(batch_index, word_index, sequence_index: int = 0)`` if batch size is greater or equal
          to 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_word_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
                the word in the sequence.
            word_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index of the word in the
                sequence.
            sequence_index (:obj:`int`, `optional`, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided word index belongs to.

        Returns:
            Optional :class:`~transformers.tokenization_utils_base.TokenSpan` Span of tokens in the encoded sequence.
            Returns :obj:`None` if no tokens correspond to the word.
        """

        raise NotImplementedError

    def token_to_chars(self, batch_or_token_index: int, token_index: Optional[int] = None) -> CharSpan:
        """
        Get the character span corresponding to an encoded token in a sequence of the batch.

        Character spans are returned as a :class:`~transformers.tokenization_utils_base.CharSpan` with:

        - **start** -- Index of the first character in the original string associated to the token.
        - **end** -- Index of the character following the last character in the original string associated to the
          token.

        Can be called as:

        - ``self.token_to_chars(token_index)`` if batch size is 1
        - ``self.token_to_chars(batch_index, token_index)`` if batch size is greater or equal to 1

        Args:
            batch_or_token_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the token in the sequence.
            token_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index of the token or tokens in
                the sequence.

        Returns:
            :class:`~transformers.tokenization_utils_base.CharSpan`: Span of characters in the original string.
        """

        raise NotImplementedError

    def char_to_token(
        self, batch_or_char_index: int, char_index: Optional[int] = None, sequence_index: int = 0
    ) -> int:
        """
        Get the index of the token in the encoded output comprising a character in the original string for a sequence
        of the batch.

        Can be called as:

        - ``self.char_to_token(char_index)`` if batch size is 1
        - ``self.char_to_token(batch_index, char_index)`` if batch size is greater or equal to 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_char_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the word in the sequence
            char_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index of the word in the
                sequence.
            sequence_index (:obj:`int`, `optional`, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided character index belongs to.


        Returns:
            :obj:`int`: Index of the token.
        """

        raise NotImplementedError

    def word_to_chars(
        self, batch_or_word_index: int, word_index: Optional[int] = None, sequence_index: int = 0
    ) -> CharSpan:
        """
        Get the character span in the original string corresponding to given word in a sequence of the batch.

        Character spans are returned as a CharSpan NamedTuple with:

        - start: index of the first character in the original string
        - end: index of the character following the last character in the original string

        Can be called as:

        - ``self.word_to_chars(word_index)`` if batch size is 1
        - ``self.word_to_chars(batch_index, word_index)`` if batch size is greater or equal to 1

        Args:
            batch_or_word_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the word in the sequence
            word_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index of the word in the
                sequence.
            sequence_index (:obj:`int`, `optional`, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided word index belongs to.

        Returns:
            :obj:`CharSpan` or :obj:`List[CharSpan]`: Span(s) of the associated character or characters in the string.
            CharSpan are NamedTuple with:

                - start: index of the first character associated to the token in the original string
                - end: index of the character following the last character associated to the token in the original
                  string
        """

        raise NotImplementedError

    def char_to_word(self, batch_or_char_index: int, char_index: Optional[int] = None, sequence_index: int = 0) -> int:
        """
        Get the word in the original string corresponding to a character in the original string of a sequence of the
        batch.

        Can be called as:

        - ``self.char_to_word(char_index)`` if batch size is 1
        - ``self.char_to_word(batch_index, char_index)`` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_char_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the character in the original string.
            char_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index of the character in the
                original string.
            sequence_index (:obj:`int`, `optional`, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided character index belongs to.


        Returns:
            :obj:`int` or :obj:`List[int]`: Index or indices of the associated encoded token(s).
        """

        raise NotImplementedError

    def convert_to_tensors(
        self, tensor_type: Optional[Union[str, TensorType]] = None, prepend_batch_axis: bool = False
    ):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                The type of tensors to use. If :obj:`str`, should be one of the values of the enum
                :class:`~transformers.file_utils.TensorType`. If :obj:`None`, no modification is done.
            prepend_batch_axis (:obj:`int`, `optional`, defaults to :obj:`False`):
                Whether or not to add the batch dimension during the conversion.
        """
        raise NotImplementedError

    @torch_required
    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        """
        Send all values to device by calling :obj:`v.to(device)` (PyTorch only).

        Args:
            device (:obj:`str` or :obj:`torch.device`): The device to put the tensors on.

        Returns:
            :class:`~transformers.BatchEncoding`: The same instance after modification.
        """

        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        raise NotImplementedError


class SpecialTokensMixin:
    """
    A mixin derived by :class:`~transformers.PreTrainedTokenizer` and :class:`~transformers.PreTrainedTokenizerFast` to
    handle specific behaviors related to special tokens. In particular, this class hold the attributes which can be
    used to directly access these special tokens in a model-independent manner and allow to set and update the special
    tokens.

    Args:
        bos_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token representing the beginning of a sentence.
        eos_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token representing the end of a sentence.
        unk_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token representing an out-of-vocabulary token.
        sep_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token separating two different sentences in the same input (used by BERT for instance).
        pad_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        cls_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token representing the class of the input (used by BERT for instance).
        mask_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token representing a masked token (used by masked-language modeling pretraining objectives, like
            BERT).
        additional_special_tokens (tuple or list of :obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A tuple or a list of additional special tokens.
    """

    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]

    def __init__(self, verbose=True, **kwargs):
        raise NotImplementedError

    def sanitize_special_tokens(self) -> int:
        """
        Make sure that all the special tokens attributes of the tokenizer (:obj:`tokenizer.mask_token`,
        :obj:`tokenizer.cls_token`, etc.) are in the vocabulary.

        Add the missing ones to the vocabulary if needed.

        Return:
            :obj:`int`: The number of tokens added in the vocabulary during the operation.
        """
        raise NotImplementedError

    def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, AddedToken]]) -> int:
        """
        Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
        special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
        current vocabulary).

        .. Note::
            When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of
            the model so that its embedding matrix matches the tokenizer.

            In order to do that, please use the :meth:`~transformers.PreTrainedModel.resize_token_embeddings` method.

        Using :obj:`add_special_tokens` will ensure your special tokens can be used in several ways:

        - Special tokens are carefully handled by the tokenizer (they are never split).
        - You can easily refer to special tokens using tokenizer class attributes like :obj:`tokenizer.cls_token`. This
          makes it easy to develop model-agnostic training and fine-tuning scripts.

        When possible, special tokens are already registered for provided pretrained models (for instance
        :class:`~transformers.BertTokenizer` :obj:`cls_token` is already registered to be :obj`'[CLS]'` and XLM's one
        is also registered to be :obj:`'</s>'`).

        Args:
            special_tokens_dict (dictionary `str` to `str` or :obj:`tokenizers.AddedToken`):
                Keys should be in the list of predefined special attributes: [``bos_token``, ``eos_token``,
                ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``,
                ``additional_special_tokens``].

                Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer
                assign the index of the ``unk_token`` to them).

        Returns:
            :obj:`int`: Number of tokens added to the vocabulary.

        Examples::

            # Let's see how to add a new classification token to GPT-2
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2Model.from_pretrained('gpt2')

            special_tokens_dict = {'cls_token': '<CLS>'}

            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print('We have added', num_added_toks, 'tokens')
            # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
            model.resize_token_embeddings(len(tokenizer))

            assert tokenizer.cls_token == '<CLS>'
        """
        raise NotImplementedError

    def add_tokens(
        self, new_tokens: Union[str, AddedToken, List[Union[str, AddedToken]]], special_tokens: bool = False
    ) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary.

        .. Note::
            When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of
            the model so that its embedding matrix matches the tokenizer.

            In order to do that, please use the :meth:`~transformers.PreTrainedModel.resize_token_embeddings` method.

        Args:
            new_tokens (:obj:`str`, :obj:`tokenizers.AddedToken` or a list of `str` or :obj:`tokenizers.AddedToken`):
                Tokens are only added if they are not already in the vocabulary. :obj:`tokenizers.AddedToken` wraps a
                string token to let you personalize its behavior: whether this token should only match against a single
                word, whether this token should strip all potential whitespaces on the left side, whether this token
                should strip all potential whitespaces on the right side, etc.
            special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Can be used to specify if the token is a special token. This mostly change the normalization behavior
                (special tokens like CLS or [MASK] are usually not lower-cased for instance).

                See details for :obj:`tokenizers.AddedToken` in HuggingFace tokenizers library.

        Returns:
            :obj:`int`: Number of tokens added to the vocabulary.

        Examples::

            # Let's see how to increase the vocabulary of Bert model and tokenizer
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
            print('We have added', num_added_toks, 'tokens')
             # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
            model.resize_token_embeddings(len(tokenizer))
        """
        raise NotImplementedError

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        raise NotImplementedError

    @property
    def bos_token(self) -> str:
        """
        :obj:`str`: Beginning of sentence token. Log an error if used while not having been set.
        """
        raise NotImplementedError

    @property
    def eos_token(self) -> str:
        """
        :obj:`str`: End of sentence token. Log an error if used while not having been set.
        """
        raise NotImplementedError

    @property
    def unk_token(self) -> str:
        """
        :obj:`str`: Unknown token. Log an error if used while not having been set.
        """
        raise NotImplementedError

    @property
    def sep_token(self) -> str:
        """
        :obj:`str`: Separation token, to separate context and query in an input sequence. Log an error if used while
        not having been set.
        """
        raise NotImplementedError

    @property
    def pad_token(self) -> str:
        """
        :obj:`str`: Padding token. Log an error if used while not having been set.
        """
        raise NotImplementedError

    @property
    def cls_token(self) -> str:
        """
        :obj:`str`: Classification token, to extract a summary of an input sequence leveraging self-attention along the
        full depth of the model. Log an error if used while not having been set.
        """
        raise NotImplementedError

    @property
    def mask_token(self) -> str:
        """
        :obj:`str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while
        not having been set.
        """
        raise NotImplementedError

    @property
    def additional_special_tokens(self) -> List[str]:
        """
        :obj:`List[str]`: All the additional special tokens you may want to use. Log an error if used while not having
        been set.
        """
        raise NotImplementedError

    @bos_token.setter
    def bos_token(self, value):
        raise NotImplementedError

    @eos_token.setter
    def eos_token(self, value):
        raise NotImplementedError

    @unk_token.setter
    def unk_token(self, value):
        raise NotImplementedError

    @sep_token.setter
    def sep_token(self, value):
        raise NotImplementedError

    @pad_token.setter
    def pad_token(self, value):
        raise NotImplementedError

    @cls_token.setter
    def cls_token(self, value):
        raise NotImplementedError

    @mask_token.setter
    def mask_token(self, value):
        raise NotImplementedError

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        raise NotImplementedError

    @property
    def bos_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the beginning of sentence token in the vocabulary. Returns :obj:`None` if the token
        has not been set.
        """
        raise NotImplementedError

    @property
    def eos_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the end of sentence token in the vocabulary. Returns :obj:`None` if the token has
        not been set.
        """
        raise NotImplementedError

    @property
    def unk_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the unknown token in the vocabulary. Returns :obj:`None` if the token has not been
        set.
        """
        raise NotImplementedError

    @property
    def sep_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the separation token in the vocabulary, to separate context and query in an input
        sequence. Returns :obj:`None` if the token has not been set.
        """
        raise NotImplementedError

    @property
    def pad_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the padding token in the vocabulary. Returns :obj:`None` if the token has not been
        set.
        """
        raise NotImplementedError

    @property
    def pad_token_type_id(self) -> int:
        """
        :obj:`int`: Id of the padding token type in the vocabulary.
        """
        raise NotImplementedError

    @property
    def cls_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the classification token in the vocabulary, to extract a summary of an input
        sequence leveraging self-attention along the full depth of the model.

        Returns :obj:`None` if the token has not been set.
        """
        raise NotImplementedError

    @property
    def mask_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the mask token in the vocabulary, used when training a model with masked-language
        modeling. Returns :obj:`None` if the token has not been set.
        """
        raise NotImplementedError

    @property
    def additional_special_tokens_ids(self) -> List[int]:
        """
        :obj:`List[int]`: Ids of all the additional special tokens in the vocabulary. Log an error if used while not
        having been set.
        """
        raise NotImplementedError

    @bos_token_id.setter
    def bos_token_id(self, value):
        raise NotImplementedError

    @eos_token_id.setter
    def eos_token_id(self, value):
        raise NotImplementedError

    @unk_token_id.setter
    def unk_token_id(self, value):
        raise NotImplementedError

    @sep_token_id.setter
    def sep_token_id(self, value):
        raise NotImplementedError

    @pad_token_id.setter
    def pad_token_id(self, value):
        raise NotImplementedError

    @cls_token_id.setter
    def cls_token_id(self, value):
        raise NotImplementedError

    @mask_token_id.setter
    def mask_token_id(self, value):
        raise NotImplementedError

    @additional_special_tokens_ids.setter
    def additional_special_tokens_ids(self, values):
        raise NotImplementedError

    @property
    def special_tokens_map(self) -> Dict[str, Union[str, List[str]]]:
        """
        :obj:`Dict[str, Union[str, List[str]]]`: A dictionary mapping special token class attributes (:obj:`cls_token`,
        :obj:`unk_token`, etc.) to their values (:obj:`'<unk>'`, :obj:`'<cls>'`, etc.).

        Convert potential tokens of :obj:`tokenizers.AddedToken` type to string.
        """
        raise NotImplementedError

    @property
    def special_tokens_map_extended(self) -> Dict[str, Union[str, AddedToken, List[Union[str, AddedToken]]]]:
        """
        :obj:`Dict[str, Union[str, tokenizers.AddedToken, List[Union[str, tokenizers.AddedToken]]]]`: A dictionary
        mapping special token class attributes (:obj:`cls_token`, :obj:`unk_token`, etc.) to their values
        (:obj:`'<unk>'`, :obj:`'<cls>'`, etc.).

        Don't convert tokens of :obj:`tokenizers.AddedToken` type to string so they can be used to control more finely
        how special tokens are tokenized.
        """
        raise NotImplementedError

    @property
    def all_special_tokens(self) -> List[str]:
        """
        :obj:`List[str]`: All the special tokens (:obj:`'<unk>'`, :obj:`'<cls>'`, etc.) mapped to class attributes.

        Convert tokens of :obj:`tokenizers.AddedToken` type to string.
        """
        raise NotImplementedError

    @property
    def all_special_tokens_extended(self) -> List[Union[str, AddedToken]]:
        """
        :obj:`List[Union[str, tokenizers.AddedToken]]`: All the special tokens (:obj:`'<unk>'`, :obj:`'<cls>'`, etc.)
        mapped to class attributes.

        Don't convert tokens of :obj:`tokenizers.AddedToken` type to string so they can be used to control more finely
        how special tokens are tokenized.
        """
        raise NotImplementedError

    @property
    def all_special_ids(self) -> List[int]:
        """
        :obj:`List[int]`: List the ids of the special tokens(:obj:`'<unk>'`, :obj:`'<cls>'`, etc.) mapped to class
        attributes.
        """
        raise NotImplementedError


ENCODE_KWARGS_DOCSTRING = r"""
            add_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to encode the sequences with the special tokens relative to their model.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`False`):
                Activates and controls padding. Accepts the following values:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            truncation (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.TruncationStrategy`, `optional`, defaults to :obj:`False`):
                Activates and controls truncation. Accepts the following values:

                * :obj:`True` or :obj:`'longest_first'`: Truncate to a maximum length specified with the argument
                  :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not
                  provided. This will truncate token by token, removing a token from the longest sequence in the pair
                  if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'only_first'`: Truncate to a maximum length specified with the argument :obj:`max_length` or to
                  the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'only_second'`: Truncate to a maximum length specified with the argument :obj:`max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`False` or :obj:`'do_not_truncate'` (default): No truncation (i.e., can output batch with
                  sequence lengths greater than the model maximum admissible input size).
            max_length (:obj:`int`, `optional`):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to :obj:`None`, this will use the predefined model maximum length if a maximum
                length is required by one of the truncation/padding parameters. If the model has no specific maximum
                input length (like XLNet) truncation/padding to a maximum length will be deactivated.
            stride (:obj:`int`, `optional`, defaults to 0):
                If set to a number along with :obj:`max_length`, the overflowing tokens returned when
                :obj:`return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
                returned to provide some overlap between truncated and overflowing sequences. The value of this
                argument defines the number of overlapping tokens.
            is_split_into_words (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to :obj:`True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
"""

ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = r"""
            return_token_type_ids (:obj:`bool`, `optional`):
                Whether to return token type IDs. If left to the default, will return the token type IDs according to
                the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are token type IDs? <../glossary.html#token-type-ids>`__
            return_attention_mask (:obj:`bool`, `optional`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are attention masks? <../glossary.html#attention-mask>`__
            return_overflowing_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return overflowing token sequences.
            return_special_tokens_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return special tokens mask information.
            return_offsets_mapping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return :obj:`(char_start, char_end)` for each token.

                This is only available on fast tokenizers inheriting from
                :class:`~transformers.PreTrainedTokenizerFast`, if using Python's tokenizer, this method will raise
                :obj:`NotImplementedError`.
            return_length  (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return the lengths of the encoded inputs.
            verbose (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to print more information and warnings.
            **kwargs: passed to the :obj:`self.tokenize()` method

        Return:
            :class:`~transformers.BatchEncoding`: A :class:`~transformers.BatchEncoding` with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.

              `What are input IDs? <../glossary.html#input-ids>`__

            - **token_type_ids** -- List of token type ids to be fed to a model (when :obj:`return_token_type_ids=True`
              or if `"token_type_ids"` is in :obj:`self.model_input_names`).

              `What are token type IDs? <../glossary.html#token-type-ids>`__

            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              :obj:`return_attention_mask=True` or if `"attention_mask"` is in :obj:`self.model_input_names`).

              `What are attention masks? <../glossary.html#attention-mask>`__

            - **overflowing_tokens** -- List of overflowing tokens sequences (when a :obj:`max_length` is specified and
              :obj:`return_overflowing_tokens=True`).
            - **num_truncated_tokens** -- Number of tokens truncated (when a :obj:`max_length` is specified and
              :obj:`return_overflowing_tokens=True`).
            - **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
              regular sequence tokens (when :obj:`add_special_tokens=True` and :obj:`return_special_tokens_mask=True`).
            - **length** -- The length of the inputs (when :obj:`return_length=True`)
"""

INIT_TOKENIZER_DOCSTRING = r"""
    Class attributes (overridden by derived classes)

        - **vocab_files_names** (:obj:`Dict[str, str]`) -- A dictionary with, as keys, the ``__init__`` keyword name of
          each vocabulary file required by the model, and as associated values, the filename for saving the associated
          file (string).
        - **pretrained_vocab_files_map** (:obj:`Dict[str, Dict[str, str]]`) -- A dictionary of dictionaries, with the
          high-level keys being the ``__init__`` keyword name of each vocabulary file required by the model, the
          low-level being the :obj:`short-cut-names` of the pretrained models with, as associated values, the
          :obj:`url` to the associated pretrained vocabulary file.
        - **max_model_input_sizes** (:obj:`Dict[str, Optinal[int]]`) -- A dictionary with, as keys, the
          :obj:`short-cut-names` of the pretrained models, and as associated values, the maximum length of the sequence
          inputs of this model, or :obj:`None` if the model has no maximum input size.
        - **pretrained_init_configuration** (:obj:`Dict[str, Dict[str, Any]]`) -- A dictionary with, as keys, the
          :obj:`short-cut-names` of the pretrained models, and as associated values, a dictionary of specific arguments
          to pass to the ``__init__`` method of the tokenizer class for this pretrained model when loading the
          tokenizer with the :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained`
          method.
        - **model_input_names** (:obj:`List[str]`) -- A list of inputs expected in the forward pass of the model.
        - **padding_side** (:obj:`str`) -- The default value for the side on which the model should have padding
          applied. Should be :obj:`'right'` or :obj:`'left'`.

    Args:
        model_max_length (:obj:`int`, `optional`):
            The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is
            loaded with :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained`, this
            will be set to the value stored for the associated model in ``max_model_input_sizes`` (see above). If no
            value is provided, will default to VERY_LARGE_INTEGER (:obj:`int(1e30)`).
        padding_side: (:obj:`str`, `optional`):
            The side on which the model should have padding applied. Should be selected between ['right', 'left'].
            Default value is picked from the class attribute of the same name.
        model_input_names (:obj:`List[string]`, `optional`):
            The list of inputs accepted by the forward pass of the model (like :obj:`"token_type_ids"` or
            :obj:`"attention_mask"`). Default value is picked from the class attribute of the same name.
        bos_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token representing the beginning of a sentence. Will be associated to ``self.bos_token`` and
            ``self.bos_token_id``.
        eos_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token representing the end of a sentence. Will be associated to ``self.eos_token`` and
            ``self.eos_token_id``.
        unk_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token representing an out-of-vocabulary token. Will be associated to ``self.unk_token`` and
            ``self.unk_token_id``.
        sep_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token separating two different sentences in the same input (used by BERT for instance). Will be
            associated to ``self.sep_token`` and ``self.sep_token_id``.
        pad_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation. Will be associated to ``self.pad_token`` and
            ``self.pad_token_id``.
        cls_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token representing the class of the input (used by BERT for instance). Will be associated to
            ``self.cls_token`` and ``self.cls_token_id``.
        mask_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token representing a masked token (used by masked-language modeling pretraining objectives, like
            BERT). Will be associated to ``self.mask_token`` and ``self.mask_token_id``.
        additional_special_tokens (tuple or list of :obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A tuple or a list of additional special tokens. Add them here to ensure they won't be split by the
            tokenization process. Will be associated to ``self.additional_special_tokens`` and
            ``self.additional_special_tokens_ids``.
"""


@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class PreTrainedTokenizerBase(SpecialTokensMixin, PushToHubMixin):
    """
    Base class for :class:`~transformers.PreTrainedTokenizer` and :class:`~transformers.PreTrainedTokenizerFast`.

    Handles shared (mostly boiler plate) methods for those two classes.
    """

    vocab_files_names: Dict[str, str] = {}
    pretrained_vocab_files_map: Dict[str, Dict[str, str]] = {}
    pretrained_init_configuration: Dict[str, Dict[str, Any]] = {}
    max_model_input_sizes: Dict[str, Optional[int]] = {}

    # first name has to correspond to main model input name
    # to make sure `tokenizer.pad(...)` works correctly
    model_input_names: List[str] = ["input_ids", "token_type_ids", "attention_mask"]
    padding_side: str = "right"
    slow_tokenizer_class = None

    def __init__(self, **kwargs):
        raise NotImplementedError

    @property
    def max_len_single_sentence(self) -> int:
        """
        :obj:`int`: The maximum length of a sentence that can be fed to the model.
        """
        raise NotImplementedError

    @property
    def max_len_sentences_pair(self) -> int:
        """
        :obj:`int`: The maximum combined length of a pair of sentences that can be fed to the model.
        """
        raise NotImplementedError

    @max_len_single_sentence.setter
    def max_len_single_sentence(self, value) -> int:
        # For backward compatibility, allow to try to setup 'max_len_single_sentence'.
        raise NotImplementedError

    @max_len_sentences_pair.setter
    def max_len_sentences_pair(self, value) -> int:
        # For backward compatibility, allow to try to setup 'max_len_sentences_pair'.
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.

        :obj:`tokenizer.get_vocab()[token]` is equivalent to :obj:`tokenizer.convert_tokens_to_ids(token)` when
        :obj:`token` is in the vocab.

        Returns:
            :obj:`Dict[str, int]`: The vocabulary.
        """
        raise NotImplementedError()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
        r"""
        Instantiate a :class:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase` (or a derived class) from
        a predefined tokenizer.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:

                - A string, the `model id` of a predefined tokenizer hosted inside a model repo.
                - A path to a `directory` containing vocabulary files required by the tokenizer, for instance saved
                  using the :meth:`~towhee.tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained`
                  method, e.g., ``./my_model_directory/``.
                - (**Deprecated**, not applicable to all derived classes) A path or url to a single saved vocabulary
                  file (if and only if the tokenizer only requires a single vocabulary file like Bert or XLNet), e.g.,
                  ``./my_model_directory/vocab.txt``.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download the vocabulary files and override the cached versions if they
                exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Attempt to resume the download if such a file
                exists.
            proxies (:obj:`Dict[str, str], `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            use_auth_token (:obj:`str` or `bool`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            subfolder (:obj:`str`, `optional`):
                In case the relevant files are located inside a subfolder of the model repo.
            inputs (additional positional arguments, `optional`):
                Will be passed along to the Tokenizer ``__init__`` method.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the Tokenizer ``__init__`` method. Can be used to set special tokens like
                ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``,
                ``mask_token``, ``additional_special_tokens``. See parameters in the ``__init__`` for more details.

        .. note::

            Passing :obj:`use_auth_token=True` is required when you want to use a private model.

        Examples::

            # We can't instantiate directly the base class `PreTrainedTokenizerBase` so let's show our examples on a derived class: BertTokenizer
            # Download vocabulary from huggingface.co and cache.
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            # Download vocabulary from huggingface.co (user-uploaded) and cache.
            tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-german-cased')

            # If vocabulary files are in a directory (e.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`)
            tokenizer = BertTokenizer.from_pretrained('./test/saved_model/')

            # If the tokenizer uses a single vocabulary file, you can point directly to this file
            tokenizer = BertTokenizer.from_pretrained('./test/saved_model/my_vocab.txt')

            # You can link tokens to special vocabulary when instantiating
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', unk_token='<unk>')
            # You should be sure '<unk>' is in the vocabulary when doing that.
            # Otherwise use tokenizer.add_special_tokens({'unk_token': '<unk>'}) instead)
            assert tokenizer.unk_token == '<unk>'

        """
        raise NotImplementedError

    @classmethod
    def _from_pretrained(
        cls, resolved_vocab_files, pretrained_model_name_or_path, init_configuration, *init_inputs, **kwargs
    ):
        # We instantiate fast tokenizers based on a slow tokenizer if we don't have access to the tokenizer.json
        # file or if `from_slow` is set to True.
        raise NotImplementedError

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> Tuple[str]:
        """
        Save the full tokenizer state.


        This method make sure the full tokenizer can then be re-loaded using the
        :meth:`~towhee.tokenization_utils_base.PreTrainedTokenizer.from_pretrained` class method.

        .. Warning::
           This won't save modifications you may have applied to the tokenizer after the instantiation (for instance,
           modifying :obj:`tokenizer.do_lower_case` after creation).

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`): The path to a directory where the tokenizer will be saved.
            legacy_format (:obj:`bool`, `optional`):
                Only applicable for a fast tokenizer. If unset (default), will save the tokenizer in the unified JSON
                format as well as in legacy format if it exists, i.e. with tokenizer specific vocabulary and a separate
                added_tokens files.

                If :obj:`False`, will only save the tokenizer in the unified JSON format. This format is incompatible
                with "slow" tokenizers (not powered by the `tokenizers` library), so the tokenizer will not be able to
                be loaded in the corresponding "slow" tokenizer.

                If :obj:`True`, will save the tokenizer in legacy format. If the "slow" tokenizer doesn't exits, a
                value error is raised.
            filename_prefix: (:obj:`str`, `optional`):
                A prefix to add to the names of the files saved by the tokenizer.
            push_to_hub (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

                .. warning::

                    Using :obj:`push_to_hub=True` will synchronize the repository you are pushing to with
                    :obj:`save_directory`, which requires :obj:`save_directory` to be a local clone of the repo you are
                    pushing to if it's an existing folder. Pass along :obj:`temp_dir=True` to use a temporary directory
                    instead.

        Returns:
            A tuple of :obj:`str`: The files saved.
        """
        raise NotImplementedError

    def _save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        file_names: Tuple[str],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        """
        Save a tokenizer using the slow-tokenizer/legacy format: vocabulary + added tokens.

        Fast tokenizers can also be saved in a unique JSON file containing {config + vocab + added-tokens} using the
        specific :meth:`~transformers.tokenization_utils_fast.PreTrainedTokenizerFast._save_pretrained`
        """
        raise NotImplementedError

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save only the vocabulary of the tokenizer (vocabulary + added tokens).

        This method won't save the configuration and special token mappings of the tokenizer. Use
        :meth:`~transformers.PreTrainedTokenizerFast._save_pretrained` to save the whole state of the tokenizer.

        Args:
            save_directory (:obj:`str`):
                The directory in which to save the vocabulary.
            filename_prefix (:obj:`str`, `optional`):
                An optional prefix to add to the named of the saved files.

        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        raise NotImplementedError

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        """
        Converts a string in a sequence of tokens, replacing unknown tokens with the :obj:`unk_token`.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.
            pair (:obj:`str`, `optional`):
                A second sequence to be encoded with the first.
            add_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to add the special tokens associated with the corresponding model.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the underlying model specific encode method. See details in
                :meth:`~transformers.PreTrainedTokenizerBase.__call__`

        Returns:
            :obj:`List[str]`: The list of tokens.
        """
        raise NotImplementedError

    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING,
        """
            **kwargs: Passed along to the `.tokenize()` method.
        """,
        """
        Returns:
            :obj:`List[int]`, :obj:`torch.Tensor`, :obj:`tf.Tensor` or :obj:`np.ndarray`: The tokenized ids of the
            text.
        """,
    )
    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> List[int]:
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Same as doing ``self.convert_tokens_to_ids(self.tokenize(text))``.

        Args:
            text (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                ``tokenize`` method) or a list of integers (tokenized string ids using the ``convert_tokens_to_ids``
                method).
            text_pair (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`, `optional`):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the ``tokenize`` method) or a list of integers (tokenized string ids using the
                ``convert_tokens_to_ids`` method).
        """
        raise NotImplementedError

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        raise NotImplementedError

    def _get_padding_truncation_strategies(
        self, padding=False, truncation=False, max_length=None, pad_to_multiple_of=None, verbose=True, **kwargs
    ):
        """
        Find the correct padding/truncation strategy with backward compatibility for old arguments (truncation_strategy
        and pad_to_max_length) and behaviors.
        """
        raise NotImplementedError

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences.

        Args:
            text (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                :obj:`is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                :obj:`is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
        """
        # Input type checking for clearer error
        def _is_valid_text_input(t):
            raise NotImplementedError

        raise NotImplementedError

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences.

        .. warning::
            This method is deprecated, ``__call__`` should be used instead.

        Args:
            text (:obj:`str`, :obj:`List[str]` or :obj:`List[int]` (the latter only for not-fast tokenizers)):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                ``tokenize`` method) or a list of integers (tokenized string ids using the ``convert_tokens_to_ids``
                method).
            text_pair (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`, `optional`):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the ``tokenize`` method) or a list of integers (tokenized string ids using the
                ``convert_tokens_to_ids`` method).
        """

        raise NotImplementedError

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        raise NotImplementedError

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.

        .. warning::
            This method is deprecated, ``__call__`` should be used instead.

        Args:
            batch_text_or_text_pairs (:obj:`List[str]`, :obj:`List[Tuple[str, str]]`, :obj:`List[List[str]]`, :obj:`List[Tuple[List[str], List[str]]]`, and for not-fast tokenizers, also :obj:`List[List[int]]`, :obj:`List[Tuple[List[int], List[int]]]`):
                Batch of sequences or pair of sequences to be encoded. This can be a list of
                string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see
                details in ``encode_plus``).
        """

        raise NotImplementedError

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        raise NotImplementedError

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with ``self.padding_side``,
        ``self.pad_token_id`` and ``self.pad_token_type_id``)

        .. note::

            If the ``encoded_inputs`` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
            result will use the same type unless you provide a different tensor type with ``return_tensors``. In the
            case of PyTorch tensors, you will lose the specific device of your tensors however.

        Args:
            encoded_inputs (:class:`~towhee.BatchEncoding`, list of :class:`~transformers.BatchEncoding`, :obj:`Dict[str, List[int]]`, :obj:`Dict[str, List[List[int]]` or :obj:`List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input (:class:`~transformers.BatchEncoding` or :obj:`Dict[str,
                List[int]]`) or a batch of tokenized inputs (list of :class:`~transformers.BatchEncoding`, `Dict[str,
                List[List[int]]]` or `List[Dict[str, List[int]]]`) so you can use this method during preprocessing as
                well as in a PyTorch Dataloader collate function.

                Instead of :obj:`List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors),
                see the note above for the return type.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask (:obj:`bool`, `optional`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are attention masks? <../glossary.html#attention-mask>`__
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            verbose (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to print more information and warnings.
        """
        raise NotImplementedError


    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed. `What are token type IDs?
        <../glossary.html#token-type-ids>`__

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (:obj:`List[int]`): The first tokenized sequence.
            token_ids_1 (:obj:`List[int]`, `optional`): The second tokenized sequence.

        Returns:
            :obj:`List[int]`: The token type ids.
        """
        raise NotImplementedError

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        This implementation does not add special tokens and this method should be overridden in a subclass.

        Args:
            token_ids_0 (:obj:`List[int]`): The first tokenized sequence.
            token_ids_1 (:obj:`List[int]`, `optional`): The second tokenized sequence.

        Returns:
            :obj:`List[int]`: The model input with special tokens.
        """
        raise NotImplementedError

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            ids (:obj:`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the ``tokenize``
                and ``convert_tokens_to_ids`` methods.
            pair_ids (:obj:`List[int]`, `optional`):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the ``tokenize``
                and ``convert_tokens_to_ids`` methods.
        """
        raise NotImplementedError

    def truncate_sequences(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Truncates a sequence pair in-place following the strategy.

        Args:
            ids (:obj:`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the ``tokenize``
                and ``convert_tokens_to_ids`` methods.
            pair_ids (:obj:`List[int]`, `optional`):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the ``tokenize``
                and ``convert_tokens_to_ids`` methods.
            num_tokens_to_remove (:obj:`int`, `optional`, defaults to 0):
                Number of tokens to remove using the truncation strategy.
            truncation_strategy (:obj:`str` or :class:`~transformers.tokenization_utils_base.TruncationStrategy`, `optional`, defaults to :obj:`False`):
                The strategy to follow for truncation. Can be:

                * :obj:`'longest_first'`: Truncate to a maximum length specified with the argument :obj:`max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate token by token, removing a token from the longest sequence in the pair if a pair of
                  sequences (or a batch of pairs) is provided.
                * :obj:`'only_first'`: Truncate to a maximum length specified with the argument :obj:`max_length` or to
                  the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'only_second'`: Truncate to a maximum length specified with the argument :obj:`max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).
            stride (:obj:`int`, `optional`, defaults to 0):
                If set to a positive number, the overflowing tokens returned will contain some tokens from the main
                sequence returned. The value of this argument defines the number of additional tokens.

        Returns:
            :obj:`Tuple[List[int], List[int], List[int]]`: The truncated ``ids``, the truncated ``pair_ids`` and the
            list of overflowing tokens.
        """
        raise NotImplementedError


    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs: Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        raise NotImplementedError


    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens in a single string. The most simple way to do it is ``" ".join(tokens)`` but we
        often want to remove sub-word tokenization artifacts at the same time.

        Args:
            tokens (:obj:`List[str]`): The token to join in a string.

        Returns:
            :obj:`str`: The joined tokens.
        """
        raise NotImplementedError

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (:obj:`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the ``__call__`` method.
            skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to clean up the tokenization spaces.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the underlying model specific decode method.

        Returns:
            :obj:`List[str]`: The list of decoded sentences.
        """
        raise NotImplementedError

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.

        Args:
            token_ids (:obj:`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the ``__call__`` method.
            skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to clean up the tokenization spaces.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the underlying model specific decode method.

        Returns:
            :obj:`str`: The decoded sentence.
        """
        raise NotImplementedError

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> str:
        raise NotImplementedError

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (:obj:`List[int]`, `optional`):
                List of ids of the second sequence.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        raise NotImplementedError

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

        Args:
            out_string (:obj:`str`): The text to clean up.

        Returns:
            :obj:`str`: The cleaned-up string.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    def _eventual_warn_about_too_long_sequence(self, ids: List[int], max_length: Optional[int], verbose: bool):
        """
        Depending on the input and internal state we might trigger a warning about a sequence that is too long for its
        corresponding model

        Args:
            ids (:obj:`List[str]`): The ids produced by the tokenization
            max_length (:obj:`int`, `optional`): The max_length desired (does not trigger a warning if it is set)
            verbose (:obj:`bool`): Whether or not to print more information and warnings.

        """
        raise NotImplementedError

    @contextmanager
    def as_target_tokenizer(self):
        """
        Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to
        sequence-to-sequence models that need a slightly different processing for the labels.
        """
        yield

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        return_tensors: str = None,
        truncation: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepare model inputs for translation. For best performance, translate one sentence at a time.

        Arguments:
            src_texts (:obj:`List[str]`):
                List of documents to summarize or source language texts.
            tgt_texts (:obj:`list`, `optional`):
                List of summaries or target language texts.
            max_length (:obj:`int`, `optional`):
                Controls the maximum length for encoder inputs (documents to summarize or source language texts) If
                left unset or set to :obj:`None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
            max_target_length (:obj:`int`, `optional`):
                Controls the maximum length of decoder inputs (target language texts or summaries) If left unset or set
                to :obj:`None`, this will use the max_length value.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`False`):
                Activates and controls padding. Accepts the following values:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            truncation (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.TruncationStrategy`, `optional`, defaults to :obj:`True`):
                Activates and controls truncation. Accepts the following values:

                * :obj:`True` or :obj:`'longest_first'`: Truncate to a maximum length specified with the argument
                  :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not
                  provided. This will truncate token by token, removing a token from the longest sequence in the pair
                  if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'only_first'`: Truncate to a maximum length specified with the argument :obj:`max_length` or to
                  the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'only_second'`: Truncate to a maximum length specified with the argument :obj:`max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`False` or :obj:`'do_not_truncate'` (default): No truncation (i.e., can output batch with
                  sequence lengths greater than the model maximum admissible input size).
            **kwargs:
                Additional keyword arguments passed along to :obj:`self.__call__`.

        Return:
            :class:`~transformers.BatchEncoding`: A :class:`~transformers.BatchEncoding` with the following fields:

            - **input_ids** -- List of token ids to be fed to the encoder.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
            - **labels** -- List of token ids for tgt_texts.

            The full set of keys ``[input_ids, attention_mask, labels]``, will only be returned if tgt_texts is passed.
            Otherwise, input_ids, attention_mask will be the only keys.
        """
        # docstyle-ignore
        formatted_warning = """
`prepare_seq2seq_batch` is deprecated and will be removed in version 5 of HuggingFace Transformers. Use the regular
`__call__` method to prepare your inputs and the tokenizer under the `as_target_tokenizer` context manager to prepare
your targets.

Here is a short example:

model_inputs = tokenizer(src_texts, ...)
with tokenizer.as_target_tokenizer():
    labels = tokenizer(tgt_texts, ...)
model_inputs["labels"] = labels["input_ids"]

See the documentation of your specific tokenizer for more details on the specific arguments to the tokenizer of choice.
For a more complete example, see the implementation of `prepare_seq2seq_batch`.
"""
        raise NotImplementedError


def get_fast_tokenizer_file(
    path_or_repo: Union[str, os.PathLike],
    revision: Optional[str] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
) -> str:
    """
    Get the tokenizer file to use for this version of transformers.

    Args:
        path_or_repo (:obj:`str` or :obj:`os.PathLike`):
            Can be either the id of a repo on huggingface.co or a path to a `directory`.
        revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id.
        use_auth_token (:obj:`str` or `bool`, `optional`):
            The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
            generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).

    Returns:
        :obj:`str`: The tokenizer file to use.
    """
    raise NotImplementedError


# To update the docstring, we need to copy the method, otherwise we change the original docstring.
PreTrainedTokenizerBase.push_to_hub = copy_func(PreTrainedTokenizerBase.push_to_hub)
PreTrainedTokenizerBase.push_to_hub.__doc__ = PreTrainedTokenizerBase.push_to_hub.__doc__.format(
    object="tokenizer", object_class="AutoTokenizer", object_files="tokenizer files"
)
