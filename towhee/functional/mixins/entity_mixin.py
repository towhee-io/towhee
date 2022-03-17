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
from typing import Dict, Any, Optional


class EntityMixin:
    """
    Mixin to help deal with Entity.
    """
    def fill_entity(self, default_kvs: Optional[Dict[str, Any]] = None, **kws):
        """
        When DataCollection's iterable exists of Entities and some indexes missing, fill default value for those indexes.

        Args:
            default_kvs (`Dict[str, Any]`):
                The key-value pairs stored in a dict.
        """
        if default_kvs:
            kws.update(default_kvs)

        for entity in self._iterable:
            for k, v in kws.items():
                if not hasattr(entity, k):
                    setattr(entity, k, v)
                    entity.register(k)

        return self.factory(self._iterable)
