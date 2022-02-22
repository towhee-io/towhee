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

from towhee.engine.operator_loader import OperatorLoader


class DispatcherMixin:
    """
    Mixin for call dispatcher for data collection
    """
    def resolve(self, call_mapping, path, *arg, **kws):
        if path in call_mapping:
            return call_mapping[path](*arg, **kws)
        else:
            loader = OperatorLoader()
            return loader.load_operator(path.replace('.', '/').replace('_', '-'), kws, tag='main')
