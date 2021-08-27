# Copyright 2020 The HuggingFace Team, the AllenNLP library authors. All rights reserved.
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
Utilities for working with the local dataset cache. Parts of this file is adapted from the AllenNLP library at
https://github.com/allenai/allennlp.
"""
import copy
import fnmatch
import functools
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import types
from collections import OrderedDict, UserDict
from contextlib import contextmanager
from dataclasses import fields
from enum import Enum
from functools import partial, wraps
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4
from zipfile import ZipFile, is_zipfile

import numpy as np
from packaging import version
from tqdm.auto import tqdm

import requests
from filelock import FileLock
from huggingface_hub import HfApi, HfFolder, Repository
from versions import importlib_metadata

from . import __version__
import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()

if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version("torch")
            logger.info(f"PyTorch version {_torch_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _torch_available = False
else:
    logger.info("Disabling PyTorch because USE_TF is set")
    _torch_available = False


if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
    _tf_available = importlib.util.find_spec("tensorflow") is not None
    if _tf_available:
        candidates = (
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tf-nightly",
            "tf-nightly-cpu",
            "tf-nightly-gpu",
            "intel-tensorflow",
            "intel-tensorflow-avx512",
            "tensorflow-rocm",
            "tensorflow-macos",
        )
        _tf_version = None
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for pkg in candidates:
            try:
                _tf_version = importlib_metadata.version(pkg)
                break
            except importlib_metadata.PackageNotFoundError:
                pass
        _tf_available = _tf_version is not None
    if _tf_available:
        if version.parse(_tf_version) < version.parse("2"):
            logger.info(f"TensorFlow found but with version {_tf_version}. Transformers requires version 2 minimum.")
            _tf_available = False
        else:
            logger.info(f"TensorFlow version {_tf_version} available.")
else:
    logger.info("Disabling Tensorflow because USE_TORCH is set")
    _tf_available = False


if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _flax_available = importlib.util.find_spec("jax") is not None and importlib.util.find_spec("flax") is not None
    if _flax_available:
        try:
            _jax_version = importlib_metadata.version("jax")
            _flax_version = importlib_metadata.version("flax")
            logger.info(f"JAX version {_jax_version}, Flax version {_flax_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _flax_available = False
else:
    _flax_available = False


_datasets_available = importlib.util.find_spec("datasets") is not None
try:
    # Check we're not importing a "datasets" directory somewhere but the actual library by trying to grab the version
    # AND checking it has an author field in the metadata that is HuggingFace.
    _ = importlib_metadata.version("datasets")
    _datasets_metadata = importlib_metadata.metadata("datasets")
    if _datasets_metadata.get("author", "") != "HuggingFace Inc.":
        _datasets_available = False
except importlib_metadata.PackageNotFoundError:
    _datasets_available = False


_faiss_available = importlib.util.find_spec("faiss") is not None
try:
    _faiss_version = importlib_metadata.version("faiss")
    logger.debug(f"Successfully imported faiss version {_faiss_version}")
except importlib_metadata.PackageNotFoundError:
    try:
        _faiss_version = importlib_metadata.version("faiss-cpu")
        logger.debug(f"Successfully imported faiss version {_faiss_version}")
    except importlib_metadata.PackageNotFoundError:
        _faiss_available = False


coloredlogs = importlib.util.find_spec("coloredlogs") is not None
try:
    _coloredlogs_available = importlib_metadata.version("coloredlogs")
    logger.debug(f"Successfully imported sympy version {_coloredlogs_available}")
except importlib_metadata.PackageNotFoundError:
    _coloredlogs_available = False


sympy_available = importlib.util.find_spec("sympy") is not None
try:
    _sympy_available = importlib_metadata.version("sympy")
    logger.debug(f"Successfully imported sympy version {_sympy_available}")
except importlib_metadata.PackageNotFoundError:
    _sympy_available = False


_keras2onnx_available = importlib.util.find_spec("keras2onnx") is not None
try:
    _keras2onnx_version = importlib_metadata.version("keras2onnx")
    logger.debug(f"Successfully imported keras2onnx version {_keras2onnx_version}")
except importlib_metadata.PackageNotFoundError:
    _keras2onnx_available = False

_onnx_available = importlib.util.find_spec("onnxruntime") is not None
try:
    _onxx_version = importlib_metadata.version("onnx")
    logger.debug(f"Successfully imported onnx version {_onxx_version}")
except importlib_metadata.PackageNotFoundError:
    _onnx_available = False


_scatter_available = importlib.util.find_spec("torch_scatter") is not None
try:
    _scatter_version = importlib_metadata.version("torch_scatter")
    logger.debug(f"Successfully imported torch-scatter version {_scatter_version}")
except importlib_metadata.PackageNotFoundError:
    _scatter_available = False


_soundfile_available = importlib.util.find_spec("soundfile") is not None
try:
    _soundfile_version = importlib_metadata.version("soundfile")
    logger.debug(f"Successfully imported soundfile version {_soundfile_version}")
except importlib_metadata.PackageNotFoundError:
    _soundfile_available = False


_timm_available = importlib.util.find_spec("timm") is not None
try:
    _timm_version = importlib_metadata.version("timm")
    logger.debug(f"Successfully imported timm version {_timm_version}")
except importlib_metadata.PackageNotFoundError:
    _timm_available = False


_torchaudio_available = importlib.util.find_spec("torchaudio") is not None
try:
    _torchaudio_version = importlib_metadata.version("torchaudio")
    logger.debug(f"Successfully imported torchaudio version {_torchaudio_version}")
except importlib_metadata.PackageNotFoundError:
    _torchaudio_available = False


torch_cache_home = os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
old_default_cache_path = os.path.join(torch_cache_home, "transformers")
# New default cache, shared with the Datasets library
hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
default_cache_path = os.path.join(hf_cache_home, "transformers")

# Onetime move from the old location to the new one if no ENV variable has been set.
if (
    os.path.isdir(old_default_cache_path)
    and not os.path.isdir(default_cache_path)
    and "PYTORCH_PRETRAINED_BERT_CACHE" not in os.environ
    and "PYTORCH_TRANSFORMERS_CACHE" not in os.environ
    and "TRANSFORMERS_CACHE" not in os.environ
):
    logger.warning(
        "In Transformers v4.0.0, the default path to cache downloaded models changed from "
        "'~/.cache/torch/transformers' to '~/.cache/huggingface/transformers'. Since you don't seem to have overridden "
        "and '~/.cache/torch/transformers' is a directory that exists, we're moving it to "
        "'~/.cache/huggingface/transformers' to avoid redownloading models you have already in the cache. You should "
        "only see this message once."
    )
    shutil.move(old_default_cache_path, default_cache_path)

PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)
SESSION_ID = uuid4().hex
DISABLE_TELEMETRY = os.getenv("DISABLE_TELEMETRY", False) in ENV_VARS_TRUE_VALUES

WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
MODEL_CARD_NAME = "modelcard.json"

SENTENCEPIECE_UNDERLINE = "▁"
SPIECE_UNDERLINE = SENTENCEPIECE_UNDERLINE  # Kept for backward compatibility

MULTIPLE_CHOICE_DUMMY_INPUTS = [
    [[0, 1, 0, 1], [1, 0, 0, 1]]
] * 2  # Needs to have 0s and 1s only since XLM uses it for langs too.
DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]

S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
CLOUDFRONT_DISTRIB_PREFIX = "https://cdn.huggingface.co"

_staging_mode = os.environ.get("HUGGINGFACE_CO_STAGING", "NO").upper() in ENV_VARS_TRUE_VALUES
_default_endpoint = "https://moon-staging.huggingface.co" if _staging_mode else "https://huggingface.co"

HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", _default_endpoint)
HUGGINGFACE_CO_PREFIX = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/{model_id}/resolve/{revision}/{filename}"

PRESET_MIRROR_DICT = {
    "tuna": "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models",
    "bfsu": "https://mirrors.bfsu.edu.cn/hugging-face-models",
}

# This is the version of torch required to run torch.fx features and torch.onnx with dictionary inputs.
TORCH_FX_REQUIRED_VERSION = version.parse("1.8")
TORCH_ONNX_DICT_INPUTS_MINIMUM_VERSION = version.parse("1.8")

_is_offline_mode = True if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False


def is_offline_mode():
    return _is_offline_mode


def is_torch_available():
    return _torch_available


def is_torch_cuda_available():
    if is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False


_torch_fx_available = _torch_onnx_dict_inputs_support_available = False
if _torch_available:
    torch_version = version.parse(importlib_metadata.version("torch"))
    _torch_fx_available = (torch_version.major, torch_version.minor) == (
        TORCH_FX_REQUIRED_VERSION.major,
        TORCH_FX_REQUIRED_VERSION.minor,
    )

    _torch_onnx_dict_inputs_support_available = torch_version >= TORCH_ONNX_DICT_INPUTS_MINIMUM_VERSION


def is_torch_fx_available():
    return _torch_fx_available


def is_torch_onnx_dict_inputs_support_available():
    return _torch_onnx_dict_inputs_support_available


def is_tf_available():
    return _tf_available


def is_coloredlogs_available():
    return _coloredlogs_available


def is_keras2onnx_available():
    return _keras2onnx_available


def is_onnx_available():
    return _onnx_available


def is_flax_available():
    return _flax_available


def is_torch_tpu_available():
    raise NotImplementedError


def is_datasets_available():
    return _datasets_available


def is_rjieba_available():
    return importlib.util.find_spec("rjieba") is not None


def is_psutil_available():
    return importlib.util.find_spec("psutil") is not None


def is_py3nvml_available():
    return importlib.util.find_spec("py3nvml") is not None


def is_apex_available():
    return importlib.util.find_spec("apex") is not None


def is_faiss_available():
    return _faiss_available


def is_scipy_available():
    return importlib.util.find_spec("scipy") is not None


def is_sklearn_available():
    if importlib.util.find_spec("sklearn") is None:
        return False
    return is_scipy_available() and importlib.util.find_spec("sklearn.metrics")


def is_sentencepiece_available():
    return importlib.util.find_spec("sentencepiece") is not None


def is_protobuf_available():
    if importlib.util.find_spec("google") is None:
        return False
    return importlib.util.find_spec("google.protobuf") is not None


def is_tokenizers_available():
    return importlib.util.find_spec("tokenizers") is not None


def is_vision_available():
    return importlib.util.find_spec("PIL") is not None


def is_in_notebook():
    try:
        # Test adapted from tqdm.autonotebook: https://github.com/tqdm/tqdm/blob/master/tqdm/autonotebook.py
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" not in get_ipython().config:
            raise ImportError("console")
        if "VSCODE_PID" in os.environ:
            raise ImportError("vscode")

        return importlib.util.find_spec("IPython") is not None
    except (AttributeError, ImportError, KeyError):
        return False


def is_scatter_available():
    return _scatter_available


def is_pandas_available():
    return importlib.util.find_spec("pandas") is not None


def is_sagemaker_dp_enabled():
    # Get the sagemaker specific env variable.
    raise NotImplementedError


def is_sagemaker_mp_enabled():
    # Get the sagemaker specific mp parameters from smp_options variable.
    raise NotImplementedError


def is_training_run_on_sagemaker():
    return "SAGEMAKER_JOB_NAME" in os.environ


def is_soundfile_availble():
    return _soundfile_available


def is_timm_available():
    return _timm_available


def is_torchaudio_available():
    return _torchaudio_available


def is_speech_available():
    # For now this depends on torchaudio but the exact dependency might evolve in the future.
    return _torchaudio_available


def torch_only_method(fn):
    def wrapper(*args, **kwargs):
        raise NotImplementedError

    raise NotImplementedError


# docstyle-ignore
DATASETS_IMPORT_ERROR = """
{0} requires the 🤗 Datasets library but it was not found in your environment. You can install it with:
```
pip install datasets
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install datasets
```
then restarting your kernel.

Note that if you have a local folder named `datasets` or a local python file named `datasets.py` in your current
working directory, python may try to import this instead of the 🤗 Datasets library. You should rename this folder or
that python file if that's the case.
"""


# docstyle-ignore
TOKENIZERS_IMPORT_ERROR = """
{0} requires the 🤗 Tokenizers library but it was not found in your environment. You can install it with:
```
pip install tokenizers
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install tokenizers
```
"""


# docstyle-ignore
SENTENCEPIECE_IMPORT_ERROR = """
{0} requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment.
"""


# docstyle-ignore
PROTOBUF_IMPORT_ERROR = """
{0} requires the protobuf library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment.
"""


# docstyle-ignore
FAISS_IMPORT_ERROR = """
{0} requires the faiss library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/facebookresearch/faiss/blob/master/INSTALL.md and follow the ones
that match your environment.
"""


# docstyle-ignore
PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
"""


# docstyle-ignore
SKLEARN_IMPORT_ERROR = """
{0} requires the scikit-learn library but it was not found in your environment. You can install it with:
```
pip install -U scikit-learn
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install -U scikit-learn
```
"""


# docstyle-ignore
TENSORFLOW_IMPORT_ERROR = """
{0} requires the TensorFlow library but it was not found in your environment. Checkout the instructions on the
installation page: https://www.tensorflow.org/install and follow the ones that match your environment.
"""


# docstyle-ignore
FLAX_IMPORT_ERROR = """
{0} requires the FLAX library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/google/flax and follow the ones that match your environment.
"""


# docstyle-ignore
SCATTER_IMPORT_ERROR = """
{0} requires the torch-scatter library but it was not found in your environment. You can install it with pip as
explained here: https://github.com/rusty1s/pytorch_scatter.
"""


# docstyle-ignore
PANDAS_IMPORT_ERROR = """
{0} requires the pandas library but it was not found in your environment. You can install it with pip as
explained here: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html.
"""


# docstyle-ignore
SCIPY_IMPORT_ERROR = """
{0} requires the scipy library but it was not found in your environment. You can install it with pip:
`pip install scipy`
"""


# docstyle-ignore
SPEECH_IMPORT_ERROR = """
{0} requires the torchaudio library but it was not found in your environment. You can install it with pip:
`pip install torchaudio`
"""

# docstyle-ignore
TIMM_IMPORT_ERROR = """
{0} requires the timm library but it was not found in your environment. You can install it with pip:
`pip install timm`
"""

# docstyle-ignore
VISION_IMPORT_ERROR = """
{0} requires the PIL library but it was not found in your environment. You can install it with pip:
`pip install pillow`
"""


BACKENDS_MAPPING = OrderedDict(
    [
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        ("faiss", (is_faiss_available, FAISS_IMPORT_ERROR)),
        ("flax", (is_flax_available, FLAX_IMPORT_ERROR)),
        ("pandas", (is_pandas_available, PANDAS_IMPORT_ERROR)),
        ("protobuf", (is_protobuf_available, PROTOBUF_IMPORT_ERROR)),
        ("scatter", (is_scatter_available, SCATTER_IMPORT_ERROR)),
        ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
        ("sklearn", (is_sklearn_available, SKLEARN_IMPORT_ERROR)),
        ("speech", (is_speech_available, SPEECH_IMPORT_ERROR)),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        ("timm", (is_timm_available, TIMM_IMPORT_ERROR)),
        ("tokenizers", (is_tokenizers_available, TOKENIZERS_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    raise NotImplementedError


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        raise NotImplementedError

    raise NotImplementedError


def add_start_docstrings_to_model_forward(*docstr):
    def docstring_decorator(fn):
        raise NotImplementedError

    raise NotImplementedError


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        raise NotImplementedError

    raise NotImplementedError


PT_RETURN_INTRODUCTION = r"""
    Returns:
        :class:`~{full_output_type}` or :obj:`tuple(torch.FloatTensor)`: A :class:`~{full_output_type}` or a tuple of
        :obj:`torch.FloatTensor` (if ``return_dict=False`` is passed or when ``config.return_dict=False``) comprising
        various elements depending on the configuration (:class:`~transformers.{config_class}`) and inputs.

"""


TF_RETURN_INTRODUCTION = r"""
    Returns:
        :class:`~{full_output_type}` or :obj:`tuple(tf.Tensor)`: A :class:`~{full_output_type}` or a tuple of
        :obj:`tf.Tensor` (if ``return_dict=False`` is passed or when ``config.return_dict=False``) comprising various
        elements depending on the configuration (:class:`~transformers.{config_class}`) and inputs.

"""


def _get_indent(t):
    """Returns the indentation in the first line of t"""
    raise NotImplementedError


def _convert_output_args_doc(output_args_doc):
    """Convert output_args_doc to display properly."""
    raise NotImplementedError


def _prepare_output_docstrings(output_type, config_class):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    raise NotImplementedError


PT_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_QUESTION_ANSWERING_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        >>> inputs = tokenizer(question, text, return_tensors='pt')
        >>> start_positions = torch.tensor([1])
        >>> end_positions = torch.tensor([3])

        >>> outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
        >>> loss = outputs.loss
        >>> start_scores = outputs.start_logits
        >>> end_scores = outputs.end_logits
"""

PT_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_MASKED_LM_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="pt")
        >>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_BASE_MODEL_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
"""

PT_MULTIPLE_CHOICE_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."
        >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

        >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors='pt', padding=True)
        >>> outputs = model(**{{k: v.unsqueeze(0) for k,v in encoding.items()}}, labels=labels)  # batch size is 1

        >>> # the linear classifier still needs to be trained
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_CAUSAL_LM_SAMPLE = r"""
    Example::

        >>> import torch
        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs, labels=inputs["input_ids"])
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": PT_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": PT_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": PT_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": PT_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": PT_MASKED_LM_SAMPLE,
    "LMHead": PT_CAUSAL_LM_SAMPLE,
    "BaseModel": PT_BASE_MODEL_SAMPLE,
}


TF_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> input_ids = inputs["input_ids"]
        >>> inputs["labels"] = tf.reshape(tf.constant([1] * tf.size(input_ids).numpy()), (-1, tf.size(input_ids))) # Batch size 1

        >>> outputs = model(inputs)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

TF_QUESTION_ANSWERING_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        >>> input_dict = tokenizer(question, text, return_tensors='tf')
        >>> outputs = model(input_dict)
        >>> start_logits = outputs.start_logits
        >>> end_logits = outputs.end_logits

        >>> all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
        >>> answer = ' '.join(all_tokens[tf.math.argmax(start_logits, 1)[0] : tf.math.argmax(end_logits, 1)[0]+1])
"""

TF_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> inputs["labels"] = tf.reshape(tf.constant(1), (-1, 1)) # Batch size 1

        >>> outputs = model(inputs)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

TF_MASKED_LM_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="tf")
        >>> inputs["labels"] = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]

        >>> outputs = model(inputs)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

TF_BASE_MODEL_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> outputs = model(inputs)

        >>> last_hidden_states = outputs.last_hidden_state
"""

TF_MULTIPLE_CHOICE_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."

        >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors='tf', padding=True)
        >>> inputs = {{k: tf.expand_dims(v, 0) for k, v in encoding.items()}}
        >>> outputs = model(inputs)  # batch size is 1

        >>> # the linear classifier still needs to be trained
        >>> logits = outputs.logits
"""

TF_CAUSAL_LM_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> outputs = model(inputs)
        >>> logits = outputs.logits
"""

TF_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": TF_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": TF_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": TF_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": TF_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": TF_MASKED_LM_SAMPLE,
    "LMHead": TF_CAUSAL_LM_SAMPLE,
    "BaseModel": TF_BASE_MODEL_SAMPLE,
}


FLAX_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')

        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
"""

FLAX_QUESTION_ANSWERING_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        >>> inputs = tokenizer(question, text, return_tensors='jax')

        >>> outputs = model(**inputs)
        >>> start_scores = outputs.start_logits
        >>> end_scores = outputs.end_logits
"""

FLAX_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')

        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
"""

FLAX_MASKED_LM_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors='jax')

        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
"""

FLAX_BASE_MODEL_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
"""

FLAX_MULTIPLE_CHOICE_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."

        >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors='jax', padding=True)
        >>> outputs = model(**{{k: v[None, :] for k,v in encoding.items()}})

        >>> logits = outputs.logits
"""

FLAX_CAUSAL_LM_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
        >>> outputs = model(**inputs)

        >>> # retrieve logts for next token
        >>> next_token_logits = outputs.logits[:, -1]
"""

FLAX_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": FLAX_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": FLAX_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": FLAX_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": FLAX_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": FLAX_MASKED_LM_SAMPLE,
    "BaseModel": FLAX_BASE_MODEL_SAMPLE,
    "LMHead": FLAX_CAUSAL_LM_SAMPLE,
}


def add_code_sample_docstrings(
    *docstr, tokenizer_class=None, checkpoint=None, output_type=None, config_class=None, mask=None, model_cls=None
):
    def docstring_decorator(fn):
        # model_class defaults to function's class if not specified otherwise
        raise NotImplementedError

    raise NotImplementedError


def replace_return_docstrings(output_type=None, config_class=None):
    def docstring_decorator(fn):
        raise NotImplementedError

    raise NotImplementedError


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def hf_bucket_url(
    model_id: str, filename: str, subfolder: Optional[str] = None, revision: Optional[str] = None, mirror=None
) -> str:
    """
    Resolve a model identifier, a file name, and an optional revision id, to a huggingface.co-hosted url, redirecting
    to Cloudfront (a Content Delivery Network, or CDN) for large files.

    Cloudfront is replicated over the globe so downloads are way faster for the end user (and it also lowers our
    bandwidth costs).

    Cloudfront aggressively caches files by default (default TTL is 24 hours), however this is not an issue here
    because we migrated to a git-based versioning system on huggingface.co, so we now store the files on S3/Cloudfront
    in a content-addressable way (i.e., the file name is its hash). Using content-addressable filenames means cache
    can't ever be stale.

    In terms of client-side caching from this library, we base our caching on the objects' ETag. An object' ETag is:
    its sha1 if stored in git, or its sha256 if stored in git-lfs. Files cached locally from transformers before v3.5.0
    are not shared with those new files, because the cached file's name contains a hash of the url (which changed).
    """
    raise NotImplementedError


def url_to_filename(url: str, etag: Optional[str] = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way. If `etag` is specified, append its hash to the url's,
    delimited by a period. If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name so that TF 2.0 can
    identify it as a HDF5 file (see
    https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    raise NotImplementedError


def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`. Raise ``EnvironmentError`` if `filename` or
    its stored metadata do not exist.
    """
    raise NotImplementedError


def get_cached_models(cache_dir: Union[str, Path] = None) -> List[Tuple]:
    """
    Returns a list of tuples representing model binaries that are cached locally. Each tuple has shape
    :obj:`(model_url, etag, size_MB)`. Filenames in :obj:`cache_dir` are use to get the metadata for each model, only
    urls ending with `.bin` are added.

    Args:
        cache_dir (:obj:`Union[str, Path]`, `optional`):
            The cache directory to search for models within. Will default to the transformers cache if unset.

    Returns:
        List[Tuple]: List of tuples each with shape :obj:`(model_url, etag, size_MB)`
    """
    raise NotImplementedError


def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    extract_compressed_file=False,
    force_extract=False,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Given something that might be a URL (or might be a local path), determine which. If it's a URL, download the file
    and cache it, and return the path to the cached file. If it's already a local path, make sure the file exists and
    then return the path

    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-download the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletely received file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
        use_auth_token: Optional string or boolean to use as Bearer token for remote files.
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and override the folder where it was extracted.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    raise NotImplementedError


def define_sagemaker_information():
    raise NotImplementedError


def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    raise NotImplementedError


def http_get(url: str, temp_file: BinaryIO, proxies=None, resume_size=0, headers: Optional[Dict[str, str]] = None):
    """
    Download remote file. Do not gobble up errors.
    """
    raise NotImplementedError


def get_from_cache(
    url: str,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Given a URL, look for the corresponding file in the local cache. If it's not there, download it. Then return the
    path to the cached file.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    raise NotImplementedError


def get_list_of_files(
    path_or_repo: Union[str, os.PathLike],
    revision: Optional[str] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
) -> List[str]:
    """
    Gets the list of files inside :obj:`path_or_repo`.

    Args:
        path_or_repo (:obj:`str` or :obj:`os.PathLike`):
            Can be either the id of a repo on huggingface.co or a path to a `directory`.
        revision (:obj:`str`, `optional`, defaults to :obj:`"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
            identifier allowed by git.
        use_auth_token (:obj:`str` or `bool`, `optional`):
            The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
            generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).

    Returns:
        :obj:`List[str]`: The list of files available in :obj:`path_or_repo`.
    """
    raise NotImplementedError


class cached_property(property):
    """
    Descriptor that mimics @property but caches output in member variable.

    From tensorflow_datasets

    Built-in in functools from Python 3.8.
    """

    def __get__(self, obj, objtype=None):
        # See docs.python.org/3/howto/descriptor.html#properties
        raise NotImplementedError


def torch_required(func):
    # Chose a different decorator name than in tests so it's clear they are not the same.
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise NotImplementedError

    raise NotImplementedError


def tf_required(func):
    # Chose a different decorator name than in tests so it's clear they are not the same.
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise NotImplementedError

    raise NotImplementedError


def is_torch_fx_proxy(x):
    raise NotImplementedError


def is_tensor(x):
    """
    Tests if ``x`` is a :obj:`torch.Tensor`, :obj:`tf.Tensor`, obj:`jaxlib.xla_extension.DeviceArray` or
    :obj:`np.ndarray`.
    """
    raise NotImplementedError


def _is_numpy(x):
    raise NotImplementedError


def _is_torch(x):
    raise NotImplementedError


def _is_torch_device(x):
    raise NotImplementedError


def _is_tensorflow(x):
    raise NotImplementedError


def _is_jax(x):
    raise NotImplementedError


def to_py_obj(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a python list.
    """
    raise NotImplementedError


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes. Otherwise behaves like a regular
    python dictionary.

    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
        method to convert it to a tuple before.
    """

    def __post_init__(self):
        raise NotImplementedError

    def __delitem__(self, *args, **kwargs):
        raise NotImplementedError

    def setdefault(self, *args, **kwargs):
        raise NotImplementedError

    def pop(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, k):
        raise NotImplementedError

    def __setattr__(self, name, value):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        raise NotImplementedError


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the ``padding`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for tab-completion
    in an IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TensorType(ExplicitEnum):
    """
    Possible values for the ``return_tensors`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, module_file, import_structure, extra_objects=None):
        raise NotImplementedError

    # Needed for autocompletion in an IDE
    def __dir__(self):
        raise NotImplementedError

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError

    def _get_module(self, module_name: str):
        raise NotImplementedError

    def __reduce__(self):
        raise NotImplementedError


def copy_func(f):
    """Returns a copy of a function f."""
    raise NotImplementedError


def is_local_clone(repo_path, repo_url):
    """
    Checks if the folder in `repo_path` is a local clone of `repo_url`.
    """
    raise NotImplementedError


class PushToHubMixin:
    """
    A Mixin containing the functionality to push a model or tokenizer to the hub.
    """

    def push_to_hub(
        self,
        repo_path_or_name: Optional[str] = None,
        repo_url: Optional[str] = None,
        use_temp_dir: bool = False,
        commit_message: Optional[str] = None,
        organization: Optional[str] = None,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> str:
        """
        Upload the {object_files} to the 🤗 Model Hub while synchronizing a local clone of the repo in
        :obj:`repo_path_or_name`.

        Parameters:
            repo_path_or_name (:obj:`str`, `optional`):
                Can either be a repository name for your {object} in the Hub or a path to a local folder (in which case
                the repository will have the name of that local folder). If not specified, will default to the name
                given by :obj:`repo_url` and a local directory with that name will be created.
            repo_url (:obj:`str`, `optional`):
                Specify this in case you want to push to an existing repository in the hub. If unspecified, a new
                repository will be created in your namespace (unless you specify an :obj:`organization`) with
                :obj:`repo_name`.
            use_temp_dir (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clone the distant repo in a temporary directory or in :obj:`repo_path_or_name` inside
                the current working directory. This will slow things down if you are making changes in an existing repo
                since you will need to clone the repo before every push.
            commit_message (:obj:`str`, `optional`):
                Message to commit while pushing. Will default to :obj:`"add {object}"`.
            organization (:obj:`str`, `optional`):
                Organization in which you want to push your {object} (you must be a member of this organization).
            private (:obj:`bool`, `optional`):
                Whether or not the repository created should be private (requires a paying subscription).
            use_auth_token (:obj:`bool` or :obj:`str`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`). Will default to
                :obj:`True` if :obj:`repo_url` is not specified.


        Returns:
            :obj:`str`: The url of the commit of your {object} in the given repository.

        Examples::

            from transformers import {object_class}

            {object} = {object_class}.from_pretrained("bert-base-cased")

            # Push the {object} to your namespace with the name "my-finetuned-bert" and have a local clone in the
            # `my-finetuned-bert` folder.
            {object}.push_to_hub("my-finetuned-bert")

            # Push the {object} to your namespace with the name "my-finetuned-bert" with no local clone.
            {object}.push_to_hub("my-finetuned-bert", use_temp_dir=True)

            # Push the {object} to an organization with the name "my-finetuned-bert" and have a local clone in the
            # `my-finetuned-bert` folder.
            {object}.push_to_hub("my-finetuned-bert", organization="huggingface")

            # Make a change to an existing repo that has been cloned locally in `my-finetuned-bert`.
            {object}.push_to_hub("my-finetuned-bert", repo_url="https://huggingface.co/sgugger/my-finetuned-bert")
        """
        raise NotImplementedError

    @staticmethod
    def _get_repo_url_from_name(
        repo_name: str,
        organization: Optional[str] = None,
        private: bool = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> str:
        raise NotImplementedError

    @classmethod
    def _create_or_get_repo(
        cls,
        repo_path_or_name: Optional[str] = None,
        repo_url: Optional[str] = None,
        organization: Optional[str] = None,
        private: bool = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> Repository:
        raise NotImplementedError

    @classmethod
    def _push_to_hub(cls, repo: Repository, commit_message: Optional[str] = None) -> str:
        raise NotImplementedError
