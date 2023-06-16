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


from towhee.utils.thirdparty.fastapi_utils import fastapi
from towhee.utils.thirdparty.uvicorn_util import uvicorn


class HTTPServer:
    """
    An HTTP server implemented based on FastAPI
    """

    def __init__(self, api_service: 'APIService'):
        self._app = fastapi.FastAPI()
        for router in api_service.routers:
            if router.methods is None:
                methods = ['POST']
            else:
                methods = router.methods if isinstance(router, list) else [router.methods]
            self._app.add_api_route(
                router.path or '/',
                router.func,
                methods=methods,
                response_model=router.output_model
            )

    @property
    def app(self):
        return self._app

    def run(self, host: str, port: int):
        uvicorn.run(self._app, host=host, port=port)
