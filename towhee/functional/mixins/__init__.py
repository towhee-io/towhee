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

from .dataset import DatasetMixin
from .dispatcher import DispatcherMixin
from .parallel import ParallelMixin
from .computer_vision import ComputerVisionMixin
from .entity_mixin import EntityMixin
from .display import DisplayMixin
from .state import StateMixin
from .metric import MetricMixin
from .ray import RayMixin


class AllMixins(DatasetMixin, DispatcherMixin, DisplayMixin, ParallelMixin,
                ComputerVisionMixin, EntityMixin, StateMixin, MetricMixin, RayMixin):

    def __init__(self) -> None:  # pylint: disable=useless-super-delegation
        super().__init__()
