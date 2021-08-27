# coding=utf-8
# Copyright 2020 Optuna, Hugging Face
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
""" Logging utilities. """

import logging
import os
import sys
import threading
from logging import CRITICAL  # NOQA
from logging import DEBUG  # NOQA
from logging import ERROR  # NOQA
from logging import FATAL  # NOQA
from logging import INFO  # NOQA
from logging import NOTSET  # NOQA
from logging import WARN  # NOQA
from logging import WARNING  # NOQA
from typing import Optional


_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING


def _get_default_logging_level():
    raise NotImplementedError


def _get_library_name() -> str:
    raise NotImplementedError


def _get_library_root_logger() -> logging.Logger:

    raise NotImplementedError


def _configure_library_root_logger() -> None:
    raise NotImplementedError


def _reset_library_root_logger() -> None:
    raise NotImplementedError


def get_log_levels_dict():
    raise NotImplementedError


def get_logger(name: Optional[str] = None) -> logging.Logger:
    raise NotImplementedError


def get_verbosity() -> int:
    raise NotImplementedError


def set_verbosity(verbosity: int) -> None:
    raise NotImplementedError


def set_verbosity_info():
    raise NotImplementedError


def set_verbosity_warning():
    raise NotImplementedError


def set_verbosity_debug():
    raise NotImplementedError


def set_verbosity_error():
    raise NotImplementedError


def disable_default_handler() -> None:
    raise NotImplementedError


def enable_default_handler() -> None:
    raise NotImplementedError


def add_handler(handler: logging.Handler) -> None:
    raise NotImplementedError


def remove_handler(handler: logging.Handler) -> None:
    raise NotImplementedError


def disable_propagation() -> None:
    raise NotImplementedError


def enable_propagation() -> None:
    raise NotImplementedError


def enable_explicit_format() -> None:
    raise NotImplementedError


def reset_format() -> None:
    raise NotImplementedError