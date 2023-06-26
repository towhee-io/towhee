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

from typing import Optional, Any, Union, List, Callable, Tuple
from functools import partial

from pydantic import BaseModel


class RouterConfig(BaseModel):
    func: Callable
    input_model: Optional[Any] = None
    output_model: Optional[Any] = None
    path: Optional[str] = None


class APIService(BaseModel):
    """
    APIService is used to define a service.
    After the definition is completed, you can use HTTPServer or the GRPCServer to started it.

    Example:
        from typing import List, Any
        from towhee import AutoPipes, api_service
        from towhee.utils.serializer import to_json

        service = api_service.APIService('Welcome')
        stn = AutoPipes.pipeline('sentence_embedding')

        @server.api(path='/embedding')
        def chat(params: List[Any]):
            return to_json(stn(*params).to_list())

        if __name__ == '__main__':
            from towhee.serve.http.server import HTTPServer
            HTTPServer(service).run('0.0.0.0', 8000)
    """
    desc: str = ''
    routers: Optional[List[RouterConfig]] = None

    def api(
            self,
            input_model: Optional[Any] = None,
            output_model: Optional[Any] = None,
            path: Optional[str] = None,
    ):

        def decorator(func: Callable):
            self.add_api(func, input_model, output_model, path)
            return func

        return decorator

    def add_api(
            self,
            func: Callable,
            input_model: Optional[Any] = None,
            output_model: Optional[Any] = None,
            path: Optional[str] = None
    ):
        if self.routers is None:
            self.routers = []

        self.routers.append(RouterConfig(
            func=func,
            input_model=input_model,
            output_model=output_model,
            path=path
        ))


def build_service(pipelines: Union[Tuple['RuntimPipeline', str], List[Tuple['RuntimPipeline', str]]],
                  desc: str = 'Welcome to use towhee pipeline service'):
    """
    Convert multiple pipelines into an APIService

    Examples:

        from towhee import AutoPipes
        from towhee import api_service

        stn = AutoPipes.pipeline('sentence_embedding')
        img = AutoPipes.pipeline('image-embedding')
        service = api_service.build_service([(stn, '/embedding/text'), (img, '/embedding/image')])

        if __name__ == '__main__':
            from towhee.serve.http.server import HTTPServer
            HTTPServer(service).run('0.0.0.0', 8000)
    """

    service = APIService(desc=desc)

    def pipe_caller(p: Callable, params: Any):
        ret = p(params)
        if hasattr(ret, 'to_list'):
            ret = ret.to_list()
        return ret

    def batch(p: Callable, params: List[Any]):
        ret_data = []
        rets = p.batch(params)
        for ret in rets:
            if hasattr(ret, 'to_list'):
                ret_data.append(ret.to_list())
            else:
                ret_data.append(ret)
        return ret_data

    if not isinstance(pipelines, list):
        pipelines = [pipelines]

    for p in pipelines:
        caller = partial(pipe_caller, p[0])
        caller.__name__ = caller.func.__name__
        caller.__doc__ = caller.func.__doc__
        caller.__code__ = caller.func.__code__
        service.add_api(caller, path=p[1])
        if hasattr(p[0], 'batch'):
            batch_caller = partial(batch, p[0])
            batch_caller.__name__ = batch_caller.func.__name__
            batch_caller.__doc__ = batch_caller.func.__doc__
            batch_caller.__code__ = batch_caller.func.__code__
            service.add_api(batch_caller, path=p[1] + '/batch')

    return service
