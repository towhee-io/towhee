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


from abc import abstractmethod


class Operator:
    """
    The base operator class.
    """

    def __init__(self):
        self._params = {}
        self.name = None
        self.executor = None
        self.is_stateless = True

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params: dict):
        # todo update parameter dict
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def op_class(cls):
        """
        Get the interface Operator class
        """
        raise NotImplementedError
