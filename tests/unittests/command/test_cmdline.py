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

# pylint: disable=consider-using-with
import os
import sys
import json
import shutil
import time
import unittest
import argparse
import requests
import subprocess

from pathlib import Path
from towhee.command.initialize import InitCommand
from towhee.serve.grpc.client import Client
from towhee.utils.serializer import from_json

PUBLIC_PATH = Path(__file__).parent.parent.resolve()
FILE_PATH = PUBLIC_PATH.parent.parent / 'towhee' / 'command' / 'cmdline.py'
PYTHON_PATH = ':'.join(sys.path)


class TestCmdline(unittest.TestCase):
    """
    Unittests for towhee cmdline.
    """

    def test_init(self):
        pyrepo = 'towhee/init-pyoperator'
        nnrepo = 'towhee/init-nnoperator'
        repo_path = PUBLIC_PATH / 'mock_operators'
        os.chdir(str(repo_path))
        args_init_pyop = argparse.Namespace(action='init', type='pyop', dir=str(repo_path / 'init-pyoperator'), uri=pyrepo, local=True)
        args_init_nnop = argparse.Namespace(action='init', type='nnop', dir=str(repo_path / 'init-nnoperator'), uri=nnrepo, local=True)

        InitCommand(args_init_pyop)()
        InitCommand(args_init_nnop)()
        self.assertTrue((repo_path / 'init-pyoperator' / 'init_pyoperator.py').is_file())
        self.assertTrue((repo_path / 'init-nnoperator' / 'init_nnoperator.py').is_file())

        shutil.rmtree(str(repo_path / 'init-pyoperator'))
        shutil.rmtree(str(repo_path / 'init-nnoperator'))

    def test_http_server(self):
        p = subprocess.Popen(
            [sys.executable, FILE_PATH, 'server', '--host', '0.0.0.0', '--port', '40001', '--python', 'main_test.py', '--http'],
            cwd=__file__.rsplit('/', 1)[0],
            env={'PYTHONPATH': PYTHON_PATH}
        )
        time.sleep(2)
        res = requests.post(url='http://0.0.0.0:40001/echo', data=json.dumps(1), timeout=3).json()
        p.terminate()

        self.assertEqual(res, 1)

    def test_grpc_server(self):
        p = subprocess.Popen(
            [sys.executable, FILE_PATH, 'server', '--host', '0.0.0.0', '--port', '50001', '--python', 'main_test.py', '--grpc'],
            cwd=__file__.rsplit('/', 1)[0],
            env={'PYTHONPATH': PYTHON_PATH}
        )
        time.sleep(2)
        grpc_client = Client(host='0.0.0.0', port=50001)
        res = grpc_client('/echo', 1)
        p.terminate()

        self.assertEqual(from_json(res.content), 1)

    def test_repo(self):
        atp = 0
        p = p = subprocess.Popen(
            [
                sys.executable,
                FILE_PATH,
                'server',
                '--host', '0.0.0.0',
                '--port', '40001',
                '--repo', 'audio-embedding', 'image-embedding',
                '--uri', '/emb/audio', '/emb/image',
                '--param', 'none', 'model_name=resnet34',
                '--http'
            ],
            cwd=__file__.rsplit('/', 1)[0],
            env={'PYTHONPATH': PYTHON_PATH}
        )

        while atp < 50:
            try:
                time.sleep(10)
                res = requests.post(
                    url='http://0.0.0.0:40001/emb/image',
                    data=json.dumps('https://github.com/towhee-io/towhee/raw/main/towhee_logo.png'),
                    timeout=None
                ).json()
                p.terminate()
                break
            except requests.exceptions.ConnectionError:
                atp += 1

        self.assertTrue(res[0][0]['_NP'])


if __name__ == '__main__':
    unittest.main()
