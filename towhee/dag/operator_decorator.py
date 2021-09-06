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


from functools import wraps
from variable_repr import VariableReprSet


def op_action(func):
    """
    Mark of the implementation of an Operator's callable part.

    Examples:
        Only interface Operator will use the @op_action mark. In most cases, method
        decorated by @op_action will be called by Operator's __call__. During an
        Operator's creation, OpMetaclass will check that method with @op_action in the
        interface Operator is overwritten by the subclass with identical input/output
        declarations.

        Example of an interface Operator

        class Img2VecOp(Operator):
            @op_action
            @abc.abstractmethod
            def forward(self, img:Image) -> FloatVector:
                pass

            @create_op_in_pipeline
            def __call__(self, img:Image):
                return forward(img)

        Example of a inherited Operator

        class VGGAnimalImg2VecOp(Img2VecOp):
            def forward(self, img:Image) -> FloatVector:
                # do the real forward works ...

        Note that Img2VecOp.forward is decorated by @op_action, and VGGAnimalImg2VecOp
        implements this function with identical input types and output types.

        See OpMetaclass._op_callable_check for more explainations.
    """
    raise NotImplementedError


def create_op_in_pipeline(func):
    """
    This is a decorator of the operator's __call__ method. It will put the operator
    into the pipeline's context -- creating a node related to the operator, and solving
    its dependency.

    Example:
        class MyOp1(Operator):
            @create_op_in_pipeline
            def __call__(self, x:int) -> NamedTuple('Outputs', [('y1', float), ('y2', str)]):
                return forward(img)

        class MyOp2(Operator):
            @create_op_in_pipeline
            def __call__(self, x:float) -> float:
                return forward(img)

        @create_pipeline
        def my_pipeline(x:int):
            op1 = MyOp1()
            op2 = MyOp2()
            y = op1(x)
            z = op2(y.y1)
            return z

    In this example, we have two operators in a pipeline, where op2 depends on the
    results of op1. Behind the scene, Towhee's compiler will create a DAG during
    the execution of my_pipeline. When the program executes the line y = op1(x),
    The decorator @create_op_in_pipeline will be called before MyOp1's __call__
    method. It will put op1 into the pipeline's context, link the pipeline's input x
    to op1's input. When the program reaches the line z = op2(y.y1), the compiler will
    add op2 into the pipeline's context, and settle the dependency between op1 and op2
    base on the fact that op2 takes op1's output as its input.
    """
    @wraps(func)
    def _create_op_in_pipeline(*args, **kwargs) -> VariableReprSet:
        """
        Solving the operator's dependency and return a VariableReprSet as the operator's
        outputs.
        """
        #raise NotImplementedError
        return func(*args, **kwargs)

    return _create_op_in_pipeline
