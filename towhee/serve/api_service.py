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
        import typing as T
        from pydantic import BaseModel
        from towhee import AutoPipes, api_service
        from towhee.serve.io import JSON

        service = api_service.APIService(desc="Welcome")
        stn = AutoPipes.pipeline('sentence_embedding')

        @service.api(path='/embedding')
        def chat(params: T.List[T.Any]):
            return stn(*params).to_list()

        class Item(BaseModel):
            url: str
            ids: T.List[int]

        @service.api(path='/echo', input_model = JSON(Item), output_model=JSON(Item))
        def echo(item: Item):
            return item

        if __name__ == '__main__':
            import sys
            if sys.argv[1] == 'http':
                from towhee.serve.http.server import HTTPServer
                HTTPServer(service).run('0.0.0.0', 8000)
            else:
                import logging
                from towhee.utils.log import engine_log
                engine_log.setLevel(logging.INFO)
                from towhee.serve.grpc.server import GRPCServer
                GRPCServer(service).run('0.0.0.0', 8000)

        Client Example:
            http:
                import requests
                requests.post('http://127.0.0.1:8000/echo', json={'url': 1, 'ids': [1, 2]}).json()
                requests.post('http://127.0.0.1:8000/embedding', json=['hello world']).json()

            grpc:
                from towhee.serve.grpc.client import Client
                c = Client('0.0.0.0', 8000)
                c('/echo', {'url': 1, 'ids': [1, 2]})
                c('/embedding', ['hello'])
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
