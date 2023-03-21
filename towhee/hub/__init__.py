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
# limitations under the License.s


from .cache_manager import CacheManager, set_local_dir
from .downloader import set_hub_url

_CACHE_MANAGER = CacheManager()


def get_operator(operator: str, tag: str, install_reqs: bool = True, latest: bool = False):
    return _CACHE_MANAGER.get_operator(operator, tag, install_reqs, latest)


__all__ = [
    'set_local_dir',
    'set_hub_url',
    'get_operator'
]
