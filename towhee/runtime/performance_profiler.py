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
import json
import time
from tabulate import tabulate
from copy import deepcopy
from pathlib import Path

from .constants import WindowAllConst
from towhee.utils.log import engine_log


class PipelineProfiler:
    """
    PipelineProfiler to trace one pipeline.
    """
    def __init__(self, dag: 'DAGRepr'):
        self.time_in = None
        self.time_out = None
        self.data = None
        self.node_tracer = {}
        self.node_report = {}
        for uid, node in dag.nodes.items():
            self.node_tracer[uid] = dict(name=node.name, iter=node.iter_info.type, init_in=[], init_out=[], queue_in=[], queue_out=[],
                                         process_in=[], process_out=[])

    def add_node_tracer(self, name, event, ts):
        ts = int(ts) / 1000000
        if event == Event.pipe_in:
            self.time_in = ts
        elif event == Event.pipe_out:
            self.time_out = ts
            self.set_node_report()
        else:
            self.node_tracer[name][event].append(ts)

    def set_node_report(self):
        self.check_tracer()  # check and set node_tracer
        for node_id, tracer in self.node_tracer.items():
            self.node_report[node_id] = dict(
                node=tracer['name'] + '(' + tracer['iter'] + ')',
                ncalls=len(tracer['process_in']),
                total_time=self.cal_time(tracer['queue_in'], tracer['queue_out']),
                init=self.cal_time(tracer['init_in'], tracer['init_out']),
                wait_data=self.cal_time(tracer['queue_in'], tracer['process_in']),
                call_op=self.cal_time(tracer['process_in'], tracer['process_out']),
                output_data=self.cal_time(tracer['process_out'], tracer['queue_out']),
            )

    def show(self):
        print('Input: ', self.data)
        print('Total time(s):', round(self.time_out - self.time_in, 3))
        headers = ['node', 'ncalls', 'total_time(s)', 'init(s)', 'wait_data(s)', 'call_op(s)', ' output_data(s)']
        print(tabulate([report.values() for _, report in self.node_report.items()], headers=headers))

    def dump(self, file_path):
        file_path = Path(file_path)
        profiler_json = self.gen_profiler_json()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(profiler_json, f)
        print(f'You can open chrome://tracing/ in your browser and load the file: {file_path}.')

    def gen_profiler_json(self, num=None):
        profiler_json = []
        if num is None:
            pipe_name = 'Pipeline'
        else:
            pipe_name = 'Pipeline:' + str(num)
        profiler_json.append({'ph': 'M', 'pid': pipe_name, 'tid': pipe_name, 'id': pipe_name, 'name': 'process_name', 'args': {'name': pipe_name}})
        profiler_json.append({'ph': 'b', 'pid': pipe_name, 'tid': pipe_name, 'id': pipe_name, 'name': pipe_name,
                              'ts': self.time_in*1000000, 'cat': 'Running Pipeline'})
        profiler_json.append({'ph': 'e', 'pid': pipe_name, 'tid': pipe_name, 'id': pipe_name, 'name': pipe_name,
                              'ts': self.time_out*1000000, 'cat': 'Running Pipeline', 'args': {'input': self.data}})

        for n_id, n_tracer in self.node_tracer.items():
            profiler_json.append({'ph': 'M', 'pid': pipe_name, 'tid': n_tracer['name'] + '(' + n_tracer['iter'] + ')', 'id': n_id + str(num),
                                  'name': n_tracer['name'], 'args': {'name': n_tracer['name']}})
            if len(n_tracer['init_in']) == 0:
                total_in = n_tracer['queue_in'][0]*1000000
            else:
                total_in = n_tracer['init_in'][0]*1000000
            profiler_json.append({'ph': 'b', 'pid': pipe_name, 'tid': n_tracer['name'] + '(' + n_tracer['iter'] + ')', 'id': n_id + str(num),
                                  'name': n_tracer['name'], 'ts': total_in, 'cat': n_tracer['name'] + '_' + n_id})
            if len(n_tracer['process_in']) == 0:
                profiler_json.append({'ph': 'e', 'pid': pipe_name, 'tid': n_tracer['name'] + '(' + n_tracer['iter'] + ')', 'id': n_id + str(num),
                                      'name': n_tracer['name'], 'ts': total_in, 'cat': n_tracer['name'] + '_' + n_id})
                continue
            profiler_json.append({'ph': 'e', 'pid': pipe_name, 'tid': n_tracer['name'] + '(' + n_tracer['iter'] + ')', 'id': n_id + str(num),
                                  'name': n_tracer['name'], 'ts': n_tracer['queue_out'][len(n_tracer['process_in'])-1] * 1000000,
                                  'cat': n_tracer['name'] + '_' + n_id})

            if len(n_tracer['init_in']) != 0:
                profiler_json.append({'ph': 'B', 'pid': pipe_name, 'tid': n_tracer['name'] + '(' + n_tracer['iter'] + ')', 'id': n_id + str(num),
                                      'name': 'init', 'ts': n_tracer['init_in'][0]*1000000})
                profiler_json.append({'ph': 'E', 'pid': pipe_name, 'tid': n_tracer['name'] + '(' + n_tracer['iter'] + ')', 'id': n_id + str(num),
                                      'name': 'init', 'ts': n_tracer['init_out'][0]*1000000})

            for i in range(len(n_tracer['process_in'])):
                profiler_json.append({'ph': 'B', 'pid': pipe_name, 'tid': n_tracer['name'] + '(' + n_tracer['iter'] + ')', 'id': n_id + str(num),
                                      'name': 'wait_data', 'ts': n_tracer['queue_in'][i] * 1000000})
                profiler_json.append({'ph': 'E', 'pid': pipe_name, 'tid': n_tracer['name'] + '(' + n_tracer['iter'] + ')', 'id': n_id + str(num),
                                      'name': 'wait_data', 'ts': n_tracer['process_in'][i] * 1000000})
                profiler_json.append({'ph': 'B', 'pid': pipe_name, 'tid': n_tracer['name'] + '(' + n_tracer['iter'] + ')', 'id': n_id + str(num),
                                      'name': 'call_op', 'ts': n_tracer['process_in'][i]*1000000})
                profiler_json.append({'ph': 'E', 'pid': pipe_name, 'tid': n_tracer['name'] + '(' + n_tracer['iter'] + ')', 'id': n_id + str(num),
                                      'name': 'call_op', 'ts': n_tracer['process_out'][i]*1000000})
                profiler_json.append({'ph': 'B', 'pid': pipe_name, 'tid': n_tracer['name'] + '(' + n_tracer['iter'] + ')', 'id': n_id + str(num),
                                      'name': 'output_data', 'ts': n_tracer['process_out'][i]*1000000})
                profiler_json.append({'ph': 'E', 'pid': pipe_name, 'tid': n_tracer['name'] + '(' + n_tracer['iter'] + ')', 'id': n_id + str(num),
                                      'name': 'output_data', 'ts': n_tracer['queue_out'][i]*1000000})
        return profiler_json

    def check_tracer(self):
        try:
            for _, node in self.node_tracer.items():
                if node['iter'] != WindowAllConst.name:
                    node['queue_in'].pop()
                assert len(node['queue_in']) >= len(node['queue_out'])
                assert len(node['process_in']) == len(node['process_out']) <= len(node['queue_in'])
                assert len(node['init_in']) == len(node['init_out'])
                assert len(node['init_in']) == 0 or len(node['init_in']) == 1
        except Exception as e:
            engine_log.error('Node:{%s} failed, please reset the tracer with `pipe.reset_tracer` and rerun it.', node['name'])
            raise e

    @staticmethod
    def cal_time(list_in, list_out):
        if len(list_in) == 0:  # concat/func is no init
            return 0
        num = min(len(list_in), len(list_out))
        total_time = sum(list_out[0:num]) - sum(list_in[0:num])
        return round(total_time, 4)


class PerformanceProfiler:
    """
    PerformanceProfiler to analysis the time profiler.
    """

    def __init__(self, time_prfilers: list, dag: 'DAGRepr'):
        self._time_prfilers = time_prfilers
        self.dag = dag
        self.timing = None
        self.pipes_profiler = []
        self.node_report = {}
        self.make_report()

    def make_report(self):
        for tf in self._time_prfilers:
            p_tracer = PipelineProfiler(self.dag)
            p_tracer.data = tf.inputs
            for ts_info in tf.time_record:
                name, event, ts = ts_info.split('::')
                p_tracer.add_node_tracer(name, event, ts)
            self.pipes_profiler.append(p_tracer)
        self.set_node_report()
        self.timing = self.get_timing_report()

    def set_node_report(self):
        self.node_report = deepcopy(self.pipes_profiler[0].node_report)
        for p_tracer in self.pipes_profiler[1:]:
            for node_id, node_tracer in p_tracer.node_report.items():
                self.node_report[node_id]['ncalls'] += node_tracer['ncalls']
                self.node_report[node_id]['total_time'] += node_tracer['total_time']
                self.node_report[node_id]['init'] += node_tracer['init']
                self.node_report[node_id]['wait_data'] += node_tracer['wait_data']
                self.node_report[node_id]['call_op'] += node_tracer['call_op']
                self.node_report[node_id]['output_data'] += node_tracer['output_data']

    def get_timing_report(self):
        timeline = self.pipes_profiler[-1].time_out - self.pipes_profiler[0].time_in
        timing_list = []
        for p_tracer in self.pipes_profiler:
            timing_list.append(p_tracer.time_out - p_tracer.time_in)
        timing_list.sort()
        total_time = sum(timing_list)
        avg_time = total_time / len(timing_list)
        return round(timeline, 4), round(avg_time, 4), round(timing_list[-1], 4), round(timing_list[0], 4)

    def show(self):
        print('Total count: ', len(self.pipes_profiler))
        print('Total time(s): ', self.timing[0])
        print('Avg time(s): ', self.timing[1])
        print('Max time(s): ', self.timing[2])
        print('Min time(s): ', self.timing[3])
        headers = ['node', 'ncalls', 'total_time(s)', 'init(s)', 'wait_data(s)', 'call_op(s)', ' output_data(s)']
        print(tabulate([report.values() for _, report in self.node_report.items()], headers=headers))

    def sort(self):
        timing_list = []
        for p_tracer in self.pipes_profiler:
            timing_list.append(p_tracer.time_out - p_tracer.time_in)
        sorted_id = sorted(range(len(timing_list)), key=lambda k: timing_list[k])
        return [self.pipes_profiler[i] for i in sorted_id]

    def max(self):
        sorted_pipe_tracer = self.sort()
        return sorted_pipe_tracer[-1]

    def __getitem__(self, item):
        return self.pipes_profiler[item]

    def dump(self, file_path):
        file_path = Path(file_path)
        profiler_json = self.gen_profiler_json()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(profiler_json, f)
        print(f'You can open chrome://tracing/ in your browser and load the file: {file_path}.')

    def gen_profiler_json(self):
        profiler_json = []
        for i, p_profiler in enumerate(self.pipes_profiler):
            profiler_json += p_profiler.gen_profiler_json(i)
        return profiler_json


class Event:
    pipe_name = '_run_pipe'
    pipe_in = 'pipe_in'
    pipe_out = 'pipe_out'
    init_in = 'init_in'
    init_out = 'init_out'
    process_in = 'process_in'
    process_out = 'process_out'
    queue_in = 'queue_in'
    queue_out = 'queue_out'


class TimeProfiler:
    """
    TimeProfiler to record the event and timestamp.
    """
    def __init__(self, enable=False):
        self._enable = enable
        self.time_record = []
        self.inputs = None

    def record(self, uid, event):
        if not self._enable:
            return
        timestamp = int(round(time.time() * 1000000))
        self.time_record.append(f'{uid}::{event}::{timestamp}')

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def reset(self):
        self.time_record = []
