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

from typing import Dict

from towhee.engine.operator_runner import runner_base
from towhee.engine.operator_runner import map_runner
from towhee.engine.operator_runner import concat_runner
from towhee.engine.operator_runner import flatmap_runner
from towhee.engine.operator_runner import filter_runner
from towhee.engine.operator_runner import window_runner
from towhee.engine.operator_runner import generator_runner
from towhee.engine.operator_io.reader import DataFrameReader
from towhee.engine.operator_io.writer import DataFrameWriter


def create_runner(
    runner_type: str,
    name: str,
    index: int,
    op_name: str,
    tag: str,
    hub_op_id: str,
    op_args: Dict[str, any],
    reader: DataFrameReader,
    writer: DataFrameWriter,
) -> runner_base.RunnerBase:
    if runner_type.lower() == 'map':
        return map_runner.MapRunner(name, index, op_name, tag, hub_op_id, op_args, reader, writer)
    elif runner_type.lower() == 'flatmap':
        return flatmap_runner.FlatMapRunner(name, index, op_name, tag, hub_op_id, op_args, reader, writer)
    elif runner_type.lower() == 'filter':
        return filter_runner.FilterRunner(name, index, op_name, tag, hub_op_id, op_args, reader, writer)
    elif runner_type.lower() == 'concat':
        return concat_runner.ConcatRunner(name, index, op_name, tag, hub_op_id, op_args, reader, writer)
    elif runner_type.lower() in ['window', 'time_window']:
        return window_runner.WindowRunner(name, index, op_name, tag, hub_op_id, op_args, reader, writer)
    elif runner_type.lower() == 'generator':
        return generator_runner.GeneratorRunner(name, index, op_name, tag, hub_op_id, op_args, reader, writer)
    else:
        raise AttributeError('No runner type named: {}'.format(runner_type))
