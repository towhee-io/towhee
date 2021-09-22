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


import threading
import time
from typing import Optional

from towhee.dataframe import DataFrame


class DfWriter(threading.Thread):
    """Put data to dataframe
    """

    def __init__(self, df: DataFrame, count: Optional[int] = None, data=()):
        super().__init__()
        self._df = df
        self._need_stop = False
        self._count = count
        self._set_sealed = False
        self._data = data

    def run(self):
        while not self._need_stop:
            if self._count is not None and self._count <= 0:
                break
            self._df.put(self._data)
            self._count -= 1
            time.sleep(0.02)

        if self._set_sealed:
            self._df.seal()

    def set_sealed_when_stop(self):
        self._set_sealed = True

    def stop(self):
        self.need_stop = True


class MultiThreadRunner:
    """
    Util for multitread test
    """

    def __init__(self, target, args, thread_num):
        self.threads = [threading.Thread(target=target, args=args)
                        for i in range(thread_num)]

    def start(self):
        for t in self.threads:
            t.start()

    def join(self):
        for t in self.threads:
            t.join()
