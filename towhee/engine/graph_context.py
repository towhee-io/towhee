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


# from towhee.engine.pipeline import Pipeline


class GraphContext:
    """
    Each row of a Pipeline's inputs will be processed individually by a pipeline, and
    each row's processing runs in a GraphContext.
    """

    def __init__(self):
        """
        Args:
            pipeline: the Pipeline this GraphContext belongs to.
        """
        self._idx = None  # the subjob index
        # todo: initialize OperatorContexts based on job.graph_repr
        self._op_contexts = None
        # self.on_start_handlers = []
        # self.on_finish_handlers = []
        # self.on_task_ready_handlers = []
        # self.on_task_start_handlers = []
        # self.on_task_finish_handlers = []
        # raise NotImplementedError

    @property
    def op_ctxs(self):
        return self._op_contexts

    def _on_start(self):
        """
        Callback before the execution of the graph.
        """
        for handler in self.on_start_handlers:
            handler(self)
        raise NotImplementedError

    def _on_finish(self):
        """
        Callback after the execution of the graph.
        """
        for handler in self.on_finish_handlers:
            handler(self)
        raise NotImplementedError
