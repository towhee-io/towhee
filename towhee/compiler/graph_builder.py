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


from graph_repr import GraphRepr
from operator_repr import OperatorRepr


class GraphBuilder:
    """
    The IntermediateGraph builder.
    
    The graph construction process is driven by OperatorMetaclass and two decorators, 
    @create_op_in_pipeline and @create_pipeline.
    """

    START_OP = '_start'
    END_OP = '_end'

    def __init__(self, builder_id:str, pipeline_func):
        self.builder_id = builder_id
        self._pipeline_func = pipeline_func
        self._op_index = 0

        self.graph_repr = GraphRepr(builder_id)
        # We add a pseudo start operator and a pseudo end operator to the graph.
    
    @property
    def start_op(self):
        """
        The start operator is a pseudo operator. The main functionality of this 
        operator is to handle the inputs of the pipeline. The start operator has no 
        inputs, and its outputs are identical to the pipeline's inputs.
        """
        if START_OP in self.graph_repr.op:
            return self.graph_repr.op[START_OP]
        else:
            # todo parse pipeline_func's inputs, and construct start operator
        raise NotImplementedError

    @property
    def end_op(self):
        """
        The start operator is a pseudo operator. The main functionality of this 
        operator is to handle the inputs of the pipeline. The start operator has no 
        inputs, and its outputs are identical to the pipeline's inputs.
        """
        if END_OP in self.graph_repr.op:
            return self.graph_repr.op[END_OP]
        else:
            # todo raise error
        raise NotImplementedError

    @end_op.setter
    def end_op(self, pipeline_outputs: VariableSet):
        """
        The end operator is a pseudo operator. The main functionality of this 
        operator is to handle the outputs of the pipeline. The end operator has no 
        outputs, and its inputs are identical to the pipeline's outputs.
        """
        if END_OP in self.graph_repr.op:
            # todo raise error
        else:
            # todo set end operator, and check *pipeline_outputs* against 
            # pipeline_func's return annotations (if any)
        raise NotImplementedError

    def add_op(self, op):
        """
        Add an OperatorRepr to the graph
        """
        op_repr = OperatorRepr(self._generate_op_name(op), op)
        # todo: setup the op_repr
        raise NotImplementedError

    def build(self) -> GraphRepr:
        """
        Build the whole graph
        """
        raise NotImplementedError

    def _generate_op_name(self, op) -> int:
        """
        Generate a unique operator name based on op.name and self._op_index
        """
        raise NotImplementedError


def get_graph_builder(key: str = None) -> GraphBuilder:
    """
    Utility for getting a graph builder

    Return: if *key* is None, then try to return the graph builder of the current 
        context, else return the builder related to the given *key*.
    """
    raise NotImplementedError


def delete_graph_builder(key: str = None):
    """
    Utility for deleting a graph builder
    """
    raise NotImplementedError

  