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
from operator import Operator
from variable_repr import VariableReprSet


class GraphBuilder:
    """
    The IntermediateGraph builder.

    The graph construction process is driven by OperatorMetaclass and two decorators,
    @create_op_in_pipeline and @create_pipeline.
    """

    START_OP = '_start'
    END_OP = '_end'

    def __init__(self, builder_id: str = None, pipeline_func=None):
        self.builder_id = builder_id
        self._pipeline_func = pipeline_func
        self._op_index = 0
        self.graph_repr = GraphRepr(builder_id)

        # We add a pseudo start operator and a pseudo end operator to the graph.
        self._start_op = OperatorRepr(GraphBuilder.START_OP)
        self._end_op = OperatorRepr(GraphBuilder.END_OP)

    @property
    def pipeline_inputs(self):
        return self._start_op.inputs

    @pipeline_inputs.setter
    def pipeline_inputs(self, inputs: VariableReprSet):
        self._start_op.inputs = inputs
        self._start_op.outputs = self._start_op.inputs

    @property
    def pipeline_outputs(self):
        return self._end_op.outputs

    @pipeline_outputs.setter
    def pipeline_outputs(self, outputs: VariableReprSet):
        self._end_op.inputs = outputs
        self._end_op.outputs = self._end_op.inputs

    def add_op(self, name: str, op: Operator) -> OperatorRepr:
        """
        Add an OperatorRepr to the graph
        """
        # op_repr = OperatorRepr(self._generate_op_name(op), op)
        # todo: setup the op_repr
        raise NotImplementedError

    def build(self) -> GraphRepr:
        """
        Build the whole graph
        """
        raise NotImplementedError

    @property
    def _start_op(self):
        """
        The start operator is a pseudo operator. The main functionality of this
        operator is to handle the inputs of the pipeline. The start operator has no
        inputs, and its outputs are identical to the pipeline's inputs.
        """
        return self.graph_repr.op[GraphBuilder.START_OP]

    @_start_op.setter
    def _start_op(self, value: OperatorRepr):
        self.graph_repr.op[GraphBuilder.START_OP] = value

    @property
    def _end_op(self):
        """
        The start operator is a pseudo operator. The main functionality of this
        operator is to handle the inputs of the pipeline. The start operator has no
        inputs, and its outputs are identical to the pipeline's inputs.
        """
        return self.graph_repr.op[GraphBuilder.END_OP]

    @_end_op.setter
    def _end_op(self, value: OperatorRepr):
        """
        The end operator is a pseudo operator. The main functionality of this
        operator is to handle the outputs of the pipeline. The end operator has no
        outputs, and its inputs are identical to the pipeline's outputs.
        """
        self.graph_repr.op[GraphBuilder.END_OP] = value


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
