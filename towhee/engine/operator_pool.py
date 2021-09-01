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
<<<<<<< HEAD
from towhee.operator.operator import Operator


class OperatorPool:
    """
    OperatorPool manages Operator creation, acquisition, release, garbage collection.
    Each TaskExecutor has one OperatorPool.
    """

    def acquire(self, name: str):
        """
        Acquire an Operator by name.
        """
        raise NotImplementedError

    def release(self, op: Operator):
        """
        Release an Operator.
        """
=======


class OperatorPool:
    """Entry to create operator
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance
            
    def acquire(self):
        raise NotImplementedError

    def release(self):
>>>>>>> 9ada236dd1ea0de151ebd569a018da7ffce5cac7
        raise NotImplementedError


