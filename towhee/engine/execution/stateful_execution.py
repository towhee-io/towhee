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


class StatefulExecution:
    """
    Execute a stateful operator
    """

    def train(self, *arg, **kws):
        self.__check_init__()
        return self._op.train(*arg, **kws)

    def fit(self, *arg):
        self._op.fit(*arg)

    @property
    def is_stateful(self):
        self.__check_init__()
        return hasattr(self._op, 'fit')

    def set_state(self, state):
        self.__check_init__()
        self._op.set_state(state)

    def set_training(self, flag):
        self._op.set_training(flag)
