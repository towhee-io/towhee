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

from .data_source import DataSourceMixin
from .dispatcher import DispatcherMixin
from .parallel import ParallelMixin
from .computer_vision import ComputerVisionMixin
from .entity_mixin import EntityMixin
from .display import DisplayMixin


class AllMixins(DataSourceMixin, DispatcherMixin, DisplayMixin, ParallelMixin,
                ComputerVisionMixin, EntityMixin):

    def __init__(self) -> None:  # pylint: disable=useless-super-delegation
        super().__init__()
