# Copyright 2023 Zilliz. All rights reserved.
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

import typing as T
import inspect
import traceback
from concurrent import futures

from towhee.utils.thirdparty.grpc_utils import grpc

from . import service_pb2
from . import service_pb2_grpc
from towhee.serve.io import IOType, JSON, TEXT

from towhee.utils.log import engine_log


class _PipelineImpl(service_pb2_grpc.PipelineServicesServicer):
    """
    Implements of grpc pipeline service.
    """

    def __init__(self, api_service: 'APIService'):
        self._router_map = {}
        for router in api_service.routers:
            self._router_map[router.path] = router

    def parse_input(self, input_model: 'IOBase', content: 'service_pb2.Content'):
        input_model = input_model or JSON()
        return input_model.from_proto(content)

    def gen_output(self, output_model, ret: T.Any):
        output_model = output_model or TEXT()
        if output_model.data_type == IOType.JSON:
            return service_pb2.Response(content=output_model.to_proto(ret),
                                        code=0, msg='Succ')
        elif output_model.data_type == IOType.TEXT:
            return service_pb2.Response(content=output_model.to_proto(ret),
                                        code=0, msg='Succ')
        return service_pb2.Response(code=-1, msg='Unkown output model')

    def _run_func(self, request):
        values = self.parse_input(self._router_map[request.path].input_model, request.content)
        func = self._router_map[request.path].func
        signature = inspect.signature(func)
        if len(signature.parameters.keys()) > 1:
            if isinstance(values, dict):
                ret = func(**values)
            else:
                ret = func(*values)
        else:
            ret = func(values)
        return self.gen_output(self._router_map[request.path].output_model, ret)

    def Predict(self, request, context):
        path = request.path
        if path not in self._router_map:
            err_msg = 'Unkown service path: %s, all paths is %s' % (path, list(self._router_map.keys()))
            engine_log.error(err_msg)
            return service_pb2.Response(code=-1, msg=err_msg)
        try:
            return self._run_func(request)
        except Exception as e:  # pylint: disable=broad-except
            engine_log.error(traceback.format_exc())
            return service_pb2.Response(code=-1, msg=str(e))


class GRPCServer:
    """
    GRPCServer
    """

    def __init__(self, api_service: 'APIService', max_workers: int = 20):
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        service_pb2_grpc.add_PipelineServicesServicer_to_server(_PipelineImpl(api_service), self._server)

    def start(self, host: str, port: int):
        self._server.add_insecure_port(str(host) + ':' + str(port))
        self._server.start()

    def run(self, host: str, port: int):
        self.start(host, port)
        self.wait()

    def wait(self):
        self._server.wait_for_termination()

    def stop(self):
        self._server.stop(None)
