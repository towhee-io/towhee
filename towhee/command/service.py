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
import sys
import importlib
import argparse

from pathlib import Path
from towhee.utils.lazy_import import LazyImport


http_server = LazyImport('http_server', globals(), 'towhee.serve.http.server')
grpc_server = LazyImport('grpc_server', globals(), 'towhee.serve.grpc.server')


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-s', '--host', default='localhost', help='The service host.')
parser.add_argument('-p', '--port', default=8000, help='The service port.')
parser.add_argument('-i', '--interface', default='service', help='The service interface, i.e. the APIService object defined in python file.')
parser.add_argument('--repo', help='Repo of the pipeline on towhee hub to start the service.')
parser.add_argument('--python', help='Path to the python file that define the pipeline.')
parser.add_argument('--http', action='store_true', help='Start service by HTTP.')
parser.add_argument('--grpc', action='store_true', help='Start service by GRPC.')


class ServerCommand:
    """
    Implementation for subcmd `towhee server`. Start pipelines as services.
    """
    def __init__(self, args):
        self._args = args

    @staticmethod
    def install(subparsers):
        subparsers.add_parser('server', parents=[parser], help='Wrap and start pipelines as services.')

    def __call__(self):
        py_path = self._args.python

        # Add module path to sys path so that module can be found
        rel_path = (Path.cwd() / py_path).resolve().parent
        abs_path = Path(py_path).resolve().parent
        sys.path.insert(0, str(abs_path))
        sys.path.insert(0, str(rel_path))

        # Load api service from target python file
        module_str = Path(py_path).stem
        attr_str = self._args.interface
        module = importlib.import_module(module_str)
        service = getattr(module, attr_str)

        if self._args.http:
            self._server = http_server.HTTPServer(service)
            self._server.run(self._args.host, int(self._args.port))
        elif self._args.grpc:
            self._server = grpc_server.GRPCServer(service)
            self._server.run(self._args.host, int(self._args.port))
