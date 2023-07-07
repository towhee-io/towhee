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
auto_pipes = LazyImport('auto_pipes', globals(), 'towhee.runtime.auto_pipes')
auto_config = LazyImport('auto_config', globals(), 'towhee.runtime.auto_config')
api_service = LazyImport('api_service', globals(), 'towhee.serve.api_service')

parser = argparse.ArgumentParser()
parser.add_argument(
    'source',
    nargs='*',
    help='The source of the pipeline, could be either in the form of `python_module:api_interface` or repository from Towhee hub.'
)
parser.add_argument('--host', default='0.0.0.0', help='The service host.')
parser.add_argument('--http-port', default=8000, help='The http service port.')
parser.add_argument('--grpc-port', help='The grpc service port.')
parser.add_argument('--uri', nargs='*', help='The uri to the pipeline service')
parser.add_argument('--params', nargs='*', help='Parameters to initialize the pipeline.')


class ServerCommand:
    """
    Implementation for subcmd `towhee server`. Start pipelines as services.
    """
    def __init__(self, args):
        self._args = args

    @staticmethod
    def install(subparsers):
        subparsers.add_parser('server', parents=[parser], help='Wrap and start pipelines as services.', add_help=False)

    def __call__(self):
        if ':' in self._args.source[0]:
            module, interface = self._args.source[0].split(':')
            file = module + '.py'
            service = self._pservice(file, interface)
        else:
            service = self._rservice(self._args.source)

        if self._args.grpc_port:
            self._server = grpc_server.GRPCServer(service)
            self._server.run(self._args.host, int(self._args.grpc_port))
        else:
            self._server = http_server.HTTPServer(service)
            self._server.run(self._args.host, int(self._args.http_port))

    def _pservice(self, file, interface):
        # Add module path to sys path so that module can be found
        rel_path = (Path.cwd() / file).resolve().parent
        abs_path = Path(file).resolve().parent
        sys.path.insert(0, str(abs_path))
        sys.path.insert(0, str(rel_path))

        # Load api service from target python file
        module_str = Path(file).stem
        attr_str = interface
        module = importlib.import_module(module_str)
        service = getattr(module, attr_str)

        return service

    def _rservice(self, repos):
        def _to_digit(x):
            try:
                float(x)
            except ValueError:
                return x

            if '.' in x:
                return float(x)
            return int(x)

        paths = self._args.uri
        params = self._args.params
        configs = [auto_config.AutoConfig.load_config(i) for i in repos]

        if params:
            for param, config in zip(params, configs):
                if param.lower() == 'none':
                    continue
                kvs = param.split(',')

                updates = {}
                for kv in kvs:
                    k, v = kv.split('=')
                    # Deal with numbers
                    updates[k] = _to_digit(v)

                config.__dict__.update(updates)

        pipes = [auto_pipes.AutoPipes.pipeline(repo, config=config) for repo, config in zip(repos, configs)]

        service = api_service.build_service(list(zip(pipes, paths)))

        return service
