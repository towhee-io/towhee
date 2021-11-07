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


from towhee.engine.operator_runner import runner_base
from towhee.engine.operator_runner import map_runner
from towhee.engine._operator_io import DataFrameReader, DataFrameWriter


def create_runner(runner_type: str, name: str, index: int,
                  reader: DataFrameReader, writer: DataFrameWriter) -> runner_base.RunnerBase:
    if runner_type.lower() == 'map':
        return map_runner.MapRunner(name, index, reader, writer)
    else:
        raise AttributeError('No runner type named: {}'.format(runner_type))
