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

import unittest
import os
from pathlib import Path
from shutil import rmtree

from towhee.hub.operator_manager import OperatorManager
from towhee.hub.pipeline_manager import PipelineManager
from towhee.utils.hub_utils import convert_dict, copy_files, init_operator, init_pipeline, generate_yaml

public_path = Path(__file__).parent.parent.resolve()


class TestHubUtils(unittest.TestCase):
    """
    Unittest for hub utils.
    """
    def test_convert_dict(self):
        d = {}
        d['test'] = '<class \'torch.Tensor\'>'
        d = convert_dict(d)

        self.assertEqual(d['test'], 'torch.Tensor')

    def test_copy_files(self):
        src = public_path / 'mock_pipelines'
        dst = public_path / 'mock_pipelines_copy'
        copy_files(public_path / 'mock_pipelines', public_path / 'mock_pipelines_copy', False)

        self.assertTrue((public_path / 'mock_pipelines_copy').is_dir())
        self.assertEqual([f.name for f in src.iterdir()], [f.name for f in dst.iterdir()])

        for src_file in src.iterdir():
            if src_file.is_dir():
                self.assertEqual([i.name for i in (src / src_file.name).iterdir()], [i.name for i in (dst / src_file.name).iterdir()])
                break

        rmtree(dst)

    def test_init_operator(self):
        py_manager = OperatorManager('towhee', 'pyoperator-template')
        nn_manager = OperatorManager('towhee', 'nnoperator-template')

        operator_dir = public_path / 'mock_operators'
        repo = ['nntemplate', 'test_nnoperator', 'pytemplate', 'test_pyoperator']
        for rp in repo:
            if (operator_dir / rp).exists():
                rmtree(operator_dir / rp)

        nn_manager.clone(local_dir=operator_dir / 'nntemplate', tag='main', install_reqs=False)
        py_manager.clone(local_dir=operator_dir / 'pytemplate', tag='main', install_reqs=False)
        init_operator(
            author='towhee', repo='test-nnoperator', is_nn=True, file_src=operator_dir / 'nntemplate', file_dst=operator_dir / 'test_nnoperator'
        )
        init_operator(
            author='towhee', repo='test-pyoperator', is_nn=False, file_src=operator_dir / 'pytemplate', file_dst=operator_dir / 'test_pyoperator'
        )

        for rp in repo:
            self.assertTrue((operator_dir / rp).is_dir())

        self.assertTrue((operator_dir / 'test_nnoperator' / 'test_nnoperator.py').is_file())
        self.assertTrue((operator_dir / 'test_nnoperator' / 'test_nnoperator.yaml').is_file())
        self.assertTrue((operator_dir / 'test_pyoperator' / 'test_pyoperator.py').is_file())
        self.assertTrue((operator_dir / 'test_pyoperator' / 'test_pyoperator.yaml').is_file())

        for rp in repo:
            if (operator_dir / rp).exists():
                rmtree(operator_dir / rp)

    def test_init_pipeline(self):
        manager = PipelineManager('towhee', 'pipeline-template')

        pipeline_dir = public_path / 'mock_pipelines'
        repo = ['template', 'test_pipeline']
        for rp in repo:
            if (pipeline_dir / rp).exists():
                rmtree(pipeline_dir / rp)

        manager.clone(local_dir=pipeline_dir / 'template', tag='main', install_reqs=False)
        init_pipeline(author='towhee', repo='test-pipeline', file_src=pipeline_dir / 'template', file_dst=pipeline_dir / 'test_pipeline')

        for rp in repo:
            self.assertTrue((pipeline_dir / rp).is_dir())

        self.assertTrue((pipeline_dir / 'test_pipeline' / 'test_pipeline.yaml').is_file())

        for rp in repo:
            if (pipeline_dir / rp).exists():
                rmtree(pipeline_dir / rp)

    def test_generate_yaml(self):
        op_dir = public_path / 'mock_operators' / 'add_operator'
        if (op_dir / 'add_operator.yaml').exists():
            os.remove(op_dir / 'add_operator.yaml')
        self.assertNotIn('add_operator.yaml', [f.name for f in op_dir.iterdir()])

        generate_yaml(author='towhee', repo='add-operator', local_dir=public_path / 'mock_operators' / 'add_operator')
        self.assertIn('add_operator.yaml', [f.name for f in op_dir.iterdir()])
        os.remove(op_dir / 'add_operator.yaml')
