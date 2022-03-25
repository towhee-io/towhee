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

from towhee.functional import DataCollection, Entity
from towhee.hparam import param_scope


def stream(iterable):
    return DataCollection.stream(iterable)


def unstream(iterable):
    return DataCollection.unstream(iterable)


def from_zip(*arg, **kws):
    return DataCollection.from_zip(*arg, **kws)


def _glob_call_back(_, index, *arg, **kws):
    if index is not None:
        return DataCollection.from_glob(
            *arg, **kws).map(lambda x: Entity(**{index: x}))
    else:
        return DataCollection.from_glob(*arg, **kws)


glob = param_scope().callholder(_glob_call_back)
