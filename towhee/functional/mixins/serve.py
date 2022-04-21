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
# pylint: disable=import-outside-toplevel

class _APIWrapper:
    """
    API Wrapper
    """
    tls = threading.local()

    def __init__(self, path='/', cls=None) -> None:
        self._queue = queue.Queue()
        self._path = path
        self._cls = cls

    def feed(self, x):
        self._queue.put(x)

    @property
    def path(self):
        return self._path

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


class ServeMixin:
    """
    Mixin for API serve

    Examples:

    # >>> from fastapi import FastAPI
    # >>> from fastapi.testclient import TestClient
    # >>> app = FastAPI()

    # >>> import towhee
    # >>> with towhee.api('/app1') as api:
    # ...     app1 = (
    # ...         api.map(lambda x: x+' -> 1')
    # ...            .map(lambda x: x+' => 1')
    # ...            .bind(app)
    # ...     )

    # >>> with towhee.api('/app2') as api:
    # ...     app2 = (
    # ...         api.map(lambda x: x+' -> 2')
    # ...            .map(lambda x: x+' => 2')
    # ...            .bind(app)
    # ...     )

    # >>> client = TestClient(app)
    # >>> client.post('/app1', '1').text
    # '"1 -> 1 => 1"'
    # >>> client.post('/app2', '2').text
    # '"2 -> 2 => 2"'
    """

    def bind(self, app=None):
        if app is None:
            from fastapi import FastAPI, Request
            app = FastAPI()
        else:
            from fastapi import Request

        api = _APIWrapper.tls.place_holder

        pipeline = _PipeWrapper(self._iterable, api)

        @app.post(api.path)
        async def wrapper(req: Request):
            nonlocal pipeline
            req = (await req.body()).decode()
            return pipeline.execute(req)

        return app

    @classmethod
    def api(cls, path='/'):
        return _APIWrapper(path, cls)


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)
