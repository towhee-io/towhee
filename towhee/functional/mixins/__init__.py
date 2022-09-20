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
from .display import DisplayMixin
from .state import StateMixin
from .metric import MetricMixin
from .ray import RayMixin
from .serve import ServeMixin
from .milvus import MilvusMixin
from .faiss import FaissMixin
from .dag import DagMixin
from .config import ConfigMixin
from .compile import CompileMixin
from .list import ListMixin
from .data_processing import DataProcessingMixin
from .stream import StreamMixin
from .safe import SafeMixin
from .format_priority import FormatPriorityMixin
from .audio import AudioMixin
from .kv_storage import KVStorageMixin


class DCMixins(DatasetMixin, DispatcherMixin, DisplayMixin, ParallelMixin,
               ComputerVisionMixin, StateMixin, MetricMixin, RayMixin,
               ServeMixin, MilvusMixin, DagMixin, FaissMixin, ConfigMixin,
               CompileMixin, ListMixin, DataProcessingMixin,
               SafeMixin, StreamMixin, FormatPriorityMixin, AudioMixin, KVStorageMixin):

    def __init__(self) -> None:  # pylint: disable=useless-super-delegation
        super().__init__()
