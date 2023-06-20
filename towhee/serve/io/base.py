# Copyright 2023 Zilliz. All rights reserved.
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


from abc import ABC, abstractmethod
from enum import Enum, auto


class IOType(Enum):
    JSON = auto()
    TEXT = auto()


class IOBase(ABC):
    """
    IOBase
    """

    def __init_subclass__(cls, *, io_type: IOType, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._data_type = io_type

    @abstractmethod
    def from_http(self, request):
        raise NotImplementedError

    @abstractmethod
    def to_http(self, obj, ctx):
        raise NotImplementedError

    @abstractmethod
    def from_proto(self, content: 'service_pb2.Content'):
        raise NotImplementedError

    @abstractmethod
    def to_proto(self, obj):
        raise NotImplementedError

    @property
    def data_type(self):
        return self._data_type
