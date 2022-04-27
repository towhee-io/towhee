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

import queue
import threading
import concurrent.futures
from towhee.functional.entity import Entity
from towhee.functional.option import Some
# pylint: disable=import-outside-toplevel


class _APIWrapper:
    """
    API Wrapper
    """
    tls = threading.local()

    def __init__(self, index=None, cls=None) -> None:
        self._queue = queue.Queue()
        self._cls = cls

        if index is not None:
            self._index = index if isinstance(index, list) else [index]
        else:
            self._index = index

    def feed(self, x) -> None:
        if self._index is None:
            entity = x
        else:
            if len(self._index) == 1:
                x = (x, )
            data = dict(zip(self._index, x))
            entity = Entity(**data)
        entity = Some(entity)
        self._queue.put(entity)

    def __iter__(self):
        while True:
            yield self._queue.get()

    def __enter__(self):
        _APIWrapper.tls.place_holder = self
        return self._cls.stream(self)

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(_APIWrapper.tls, 'place_holder'):
            _APIWrapper.tls.place_holder = None


class _PipeWrapper:
    """
    Wrapper for execute pipeline as function
    """

    def __init__(self, pipe, place_holder) -> None:
        self._pipe = pipe
        self._place_holder = place_holder
        self._futures = queue.Queue()
        self._lock = threading.Lock()
        self._executor = threading.Thread(target=self.worker, daemon=True)
        self._executor.start()

    def worker(self):
        while True:
            future = self._futures.get()
            result = next(self._pipe)
            future.set_result(result)

    def execute(self, x):
        with self._lock:
            future = concurrent.futures.Future()
            self._futures.put(future)
            self._place_holder.feed(x)
        return future.result()


async def _decode_content(req):
    from multipart.multipart import parse_options_header
    content_type_header = req.headers.get('Content-Type')
    content_type, _ = parse_options_header(content_type_header)

    if content_type in {b'multipart/form-data'}:
        return await req.form()
    if content_type.startswith(b'image/'):
        return await req.body()
    return (await req.body()).decode()


class ServeMixin:
    """
    Mixin for API serve
    """

    def serve(self, path='/', app=None):
        """
        Serve the DataCollection as a RESTful API

        Args:
            path (str, optional): API path. Defaults to '/'.
            app (_type_, optional): The FastAPI app the API bind to, will create one if None.

        Returns:
            _type_: the app that bind to

        Examples:

        >>> from fastapi import FastAPI
        >>> from fastapi.testclient import TestClient
        >>> app = FastAPI()

        >>> import towhee
        >>> with towhee.api() as api:
        ...     app1 = (
        ...         api.map(lambda x: x+' -> 1')
        ...            .map(lambda x: x+' => 1')
        ...            .serve('/app1', app)
        ...     )

        >>> with towhee.api['x']() as api:
        ...     app2 = (
        ...         api.runas_op['x', 'x_plus_1'](func=lambda x: x+' -> 2')
        ...            .runas_op['x_plus_1', 'y'](func=lambda x: x+' => 2')
        ...            .select['y']()
        ...            .serve('/app2', app)
        ...     )

        >>> with towhee.api() as api:
        ...     app2 = (
        ...         api.parse_json()
        ...            .runas_op['x', 'x_plus_1'](func=lambda x: x+' -> 3')
        ...            .runas_op['x_plus_1', 'y'](func=lambda x: x+' => 3')
        ...            .select['y']()
        ...            .serve('/app3', app)
        ...     )

        >>> client = TestClient(app)
        >>> client.post('/app1', '1').text
        '"1 -> 1 => 1"'
        >>> client.post('/app2', '2').text
        '{"y":"2 -> 2 => 2"}'
        >>> client.post('/app3', '{"x": "3"}').text
        '{"y":"3 -> 3 => 3"}'
        """
        if app is None:
            from fastapi import FastAPI, Request
            app = FastAPI()
        else:
            from fastapi import Request

        api = _APIWrapper.tls.place_holder

        pipeline = _PipeWrapper(self._iterable, api)

        @app.post(path)
        async def wrapper(req: Request):
            nonlocal pipeline
            req = await _decode_content(req)
            rsp = pipeline.execute(req)
            if rsp.is_empty():
                return rsp.get()
            return rsp.get()

        return app

    def as_function(self):
        """
        Make the DataCollection as callable function

        Returns:
            _type_: a callable function

        Examples:

        >>> import towhee
        >>> with towhee.api() as api:
        ...     func1 = (
        ...         api.map(lambda x: x+' -> 1')
        ...            .map(lambda x: x+' => 1')
        ...            .as_function()
        ...     )

        >>> with towhee.api['x']() as api:
        ...     func2 = (
        ...         api.runas_op['x', 'x_plus_1'](func=lambda x: x+' -> 2')
        ...            .runas_op['x_plus_1', 'y'](func=lambda x: x+' => 2')
        ...            .select['y']()
        ...            .as_raw()
        ...            .as_function()
        ...     )

        >>> with towhee.api() as api:
        ...     func3 = (
        ...         api.parse_json()
        ...            .runas_op['x', 'x_plus_1'](func=lambda x: x+' -> 3')
        ...            .runas_op['x_plus_1', 'y'](func=lambda x: x+' => 3')
        ...            .select['y']()
        ...            .as_json()
        ...            .as_function()
        ...     )

        >>> func1('1')
        '1 -> 1 => 1'
        >>> func2('2')
        '2 -> 2 => 2'
        >>> func3('{"x": "3"}')
        '{"y": "3 -> 3 => 3"}'
        """

        api = _APIWrapper.tls.place_holder

        pipeline = _PipeWrapper(self._iterable, api)

        def wrapper(req):
            rsp = pipeline.execute(req)
            if rsp.is_empty():
                return rsp.get()
            return rsp.get()

        return wrapper

    @classmethod
    def api(cls, index=None):
        return _APIWrapper(index=index, cls=cls)
