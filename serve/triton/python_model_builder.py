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

from typing import List, Tuple, Any

import serve.triton.type_gen as tygen
import serve.triton.format_utils as fmt


class PyModelBuilder:
    """
    The base class of python model builder.
    Subclass of PyModelBuilder should implement the following methods:
    `gen_import`, `gen_init`.
    """

    def gen_imports(self):
        raise NotImplementedError('gen_imports not implemented')

    def gen_initialize(self):
        raise NotImplementedError('gen_initialize not implemented')

    def gen_execute(self):
        raise NotImplementedError('gen_execute not implemented')

    def build(self, save_path: str = './model.py'):
        with open(save_path, 'wt') as f:
            f.writelines(self.gen_imports())
            f.writelines(self.gen_blank_lines(2))
            f.writelines(self.gen_class_name())
            f.writelines(self.gen_blank_lines(1))
            f.writelines(fmt.intend(self.gen_initialize()))

    @staticmethod
    def gen_class_name():
        lines = ['class TritonPythonModel:']
        return fmt.add_line_separator(lines)

    @staticmethod
    def gen_blank_lines(n=1):
        lines = [''] * n
        return fmt.add_line_separator(lines)

    @staticmethod
    def _from_tensor_to_obj(
        type_info: tygen.TypeInfo,
        obj_init_code: str,
        obj_var: str,
        tensor_vars: List[str]
    ):
        """
        Generate the codes converting Tensor to python object
        """
        trs_placeholders = [attr.tensor_placeholder for attr in type_info.attr_info]

        # check existence of placeholders
        for tr in trs_placeholders:
            if obj_init_code.find(tr) == -1:
                raise ValueError('Can not match placeholder %s with init function %s.' % (tr, trs_placeholders))

        if type_info.is_list:
            init_args = ['arg' + str(i) for i in range(len(trs_placeholders))]
            for tr, arg in zip(trs_placeholders, init_args):
                obj_init_code = obj_init_code.replace(tr, arg)

            line = obj_var + ' = [' + obj_init_code + ' for ' + ', '.join(init_args) + ' in zip(' + ', '.join(tensor_vars) + ')]'
        else:
            for tr, arg in zip(trs_placeholders, tensor_vars):
                obj_init_code = obj_init_code.replace(tr, arg)

            line = obj_var + ' = ' + obj_init_code

        return [line]

    @staticmethod
    def _from_obj_to_tensor(
        type_info: tygen.TypeInfo,
        obj_var: str,
        tensor_vars: List[str],
        tensor_names: List[str],
        obj_placeholder='$obj'
    ):
        """
        Generate the codes converting python object to Tensor
        """

        lines = []

        for attr, tr_name, tr_var in zip(type_info.attr_info, tensor_names, tensor_vars):
            data = attr.obj_placeholder.replace(obj_placeholder, obj_var)
            dtype = attr.numpy_dtype
            ndarray = 'numpy.array(' + data + ', ' + dtype + ')'
            line = tr_var + ' = pb_utils.Tensor(\'' + tr_name + '\', ' + ndarray + ')'
            lines.append(line)

        return lines


class PickledCallablePyModelBuilder(PyModelBuilder):
    """
    Build python model from pickled
    """

    def __init__(
        self,
        module_name: str,
        callable_name: str,
        python_file_path: str,
        pickle_file_path: str,
        input_annotations: List[Tuple[Any, Tuple]],
        output_annotations: List[Tuple[Any, Tuple]]
    ):
        self.module_name = module_name
        self.callable_name = callable_name
        self.python_file_path = python_file_path
        self.pickle_file_path = pickle_file_path
        self.input_annotations = input_annotations
        self.output_annotations = output_annotations

    def gen_imports(self):
        lines = []
        lines.append('import inspect')
        lines.append('import pickle')
        lines.append('import importlib')
        lines.append('import sys')
        lines.append('import triton_python_backend_utils as pb_utils')

        return fmt.add_line_separator(lines)

    def gen_initialize(self):
        lines = []
        lines.append('def initialize(self, args):')
        lines.append('')
        lines.append('# load module')
        lines.append('spec = importlib.util.spec_from_file_location(\'' + self.module_name + '\', \'' + self.python_file_path + '\')')
        lines.append('module = importlib.util.module_from_spec(spec)')
        lines.append('sys.modules[\'' + self.module_name + '\'] = module')
        lines.append('spec.loader.exec_module(module)')
        lines.append('')
        lines.append('# create callable object')
        lines.append('callable_cls = ' + self.module_name + '.' + self.callable_name)
        lines.append('with open(\'' + self.pickle_file_path + '\', \'rb\') as f:')
        lines.append(fmt.intend('self.callable_obj = pickle.load(f)'))

        lines = lines[:1] + fmt.intend(lines[1:])
        return fmt.add_line_separator(lines)

#    def gen_execute(self):
#        lines = []
#        lines.append('def execute(self, requests):')
#        lines.append('')
#        lines.append('responses = []')
#
#        taskloop = []
#        taskloop.append('for request in requests:')
#        taskloop.append('# get inputs from request')
#        attrs = []
#        for type_info in self.input_annotations:
#            attrs = attrs + type_info.
#
#        return fmt.add_line_separator(lines)


class OpPyModelBuilder:
    pass


def gen_model_from_pickled_callable():
    pass


def gen_model_from_op():
    pass
