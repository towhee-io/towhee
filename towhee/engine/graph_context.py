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


from towhee.engine.pipeline import Pipeline


class GraphContext:
    """
    Each row of a Pipeline's inputs will be processed individually by a pipeline, and 
    each row's processing runs in a GraphContext.
    """

    def __init__(self, pipeline):
        """
        Args:
            pipeline: the Pipeline this GraphContext belongs to.
        """
        self._job = job
        self.idx = None # the subjob index
        # todo: initialize OperatorContexts based on job.graph_repr
        self.op_contexts = None
        raise NotImplementedError
 
    def on_start(self, handler: function):
        """
        Set a custom handler that called before the execution of the graph.
        """
        self._on_start_handler = handler
        raise NotImplementedError

    def on_finish(self, handler: function):
        """
        Set a custom handler that called after the execution of the graph.
        """
        self._on_finish_handler = handler
        raise NotImplementedError