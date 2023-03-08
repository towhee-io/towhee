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
# WITHOUT_ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class MapConst:
    name = 'map'


class FilterConst:
    class _Param:
        filter_by ='filter_by'

    name = 'filter'
    param = _Param()


class WindowConst:
    class _Param:
        step = 'step'
        size = 'size'

    name = 'window'
    param = _Param()


class TimeWindowConst:
    class _Param:
        time_range_sec = 'time_range_sec'
        time_step_sec = 'time_step_sec'
        timestamp_col = 'timestamp_col'

    name = 'time_window'
    param = _Param()


class WindowAllConst:
    name = 'window_all'


class ConcatConst:
    name = 'concat'


class FlatMapConst:
    name = 'flat_map'


class InputConst:
    name = '_input'


class OutputConst:
    name = '_output'


class OPType:
    HUB = 'hub'
    LAMBDA = 'lambda'
    CALLABLE = 'callable'
