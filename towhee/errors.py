# Copyright 2021 Zilliz. All rights reserved.
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


class TowheeError(Exception):
    """
    Towhee exception base.
    """


class GraphError(TowheeError):  # pylint: disable=empty-docstring
    """
    """


class OpIOTypeError(TowheeError):  # pylint: disable=empty-docstring
    """
    """


class OpFailedError(TowheeError):  # pylint: disable=empty-docstring
    """
    """


class OpTypeError(TowheeError):  # pylint: disable=empty-docstring
    """
    """


class SchedulerTypeError(TowheeError):  # pylint: disable=empty-docstring
    """
    """


class NoSchedulerError(TowheeError):  # pylint: disable=empty-docstring
    """
    """


class EmptyInputError(TowheeError):  # pylint: disable=empty-docstring
    """
    """
