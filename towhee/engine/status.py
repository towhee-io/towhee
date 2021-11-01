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


class Status:
    """
    Return status
    """

    def __init__(self, code: int, msg: str = None, data: any = None):
        self._code = code
        self._msg = msg
        self._data = data

    @staticmethod
    def ok_status(data: any = None) -> 'Status':
        return Status(0, data=data)

    @staticmethod
    def err_status(msg: str) -> 'Status':
        return Status(-1, msg=msg)

    @property
    def msg(self):
        return self._msg

    @property
    def data(self):
        return self._data

    def is_ok(self):
        return self._code == 0
