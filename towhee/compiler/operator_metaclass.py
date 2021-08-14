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
    return func


class OperatorMetaclass(type):
    '''
    Metaclass for creating Operators. 
    '''

    def _op_callable_check(self, op_dict):
        """
        Test whether an Operator can be called.

        Args:

        Returns:
            True if the Operator follows all the callable rules, False other wise.

        Raises:

        Examples:
            In Towhee, each Operator should be a callable. Each interface Operator will 
            define an abstract method specifying its functionality. For example,

            class Img2VecOp(Operator):
                @op_action
                @abc.abstractmethod
                def forward(self, img:Image) -> FloatVector:
                    pass

                def __call__(self, img:Image):
                    return forward(img)


            // in pytorch
            m.forward(xxx)
            m(xxx)

            // in towhee, pipeline definition
            op = XXXFaceImg2Vec(init_params)
            result = op(xxx)

            The forward function plays as the core functionality of Img2VecOp, and it 
            will be called by __call__. Note that the function is decorated by 
            @op_action. _op_callable_check will go through Img2VecOp's __dict__, and 
            find out the methods with @op_action decoration. For each decorated 
            function, OpMetaclass will check and makesure that it is overwritten by 
            the subclass with identical input/output declarations.

            Here is an example class inherited Img2VecOp.

            class VGGAnimalImg2VecOp(Img2VecOp):
                def forward(self, img:Image) -> FloatVector:
                    # do the real forward works ...

            The strongly typed override mechanism guarantees that the inputs and outputs
            standard of a interface Operator is strictly followed by its subclasses. 
            If not, the OpMetaclass will find such inconsistency during the subclass's 
            creation time.

            An Operator may have multiple outputs. For this case, the interface
            Operator should define the outputs as a NamedTuple. For example,

            class ObjectDetectionOp(Operator):
                @op_action
                @abc.abstractmethod
                def inference(self, img:Image) -> NamedTuple('Outputs', [('bboxes', list[BBox]), ('labels', list[str]), ('scores', list[float])]):
                    pass

                def __call__(self, img:Image):
                    return inference(img)
        """
        raise NotImplementedError