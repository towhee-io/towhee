# Copyright 2022 Zilliz. All rights reserved.
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

import importlib

import importlib_metadata
from towhee.utils.log import trainer_log

_captum_available = importlib.util.find_spec("captum") is not None
try:
    _captum_version = importlib_metadata.version("captum")
    trainer_log.info("Successfully imported captum version %s", _captum_version)
except importlib_metadata.PackageNotFoundError:
    _captum_available = False

_tensorboard_available = importlib.util.find_spec("tensorboard") is not None \
                         or importlib.util.find_spec("tensorboardX") is not None
try:
    from torch.utils.tensorboard import SummaryWriter  # pylint: disable=import-outside-toplevel

    trainer_log.info(SummaryWriter)
except (ImportError, AttributeError):
    _tensorboard_available = False


def is_captum_available():
    return _captum_available


def is_tensorboard_available():
    return _tensorboard_available


def is_matplotlib_available():
    return importlib.util.find_spec("matplotlib") is not None
