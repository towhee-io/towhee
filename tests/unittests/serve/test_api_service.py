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

from typing import List
import unittest

from towhee import api_service

class TestAPIService(unittest.TestCase):
    """
    Test api service
    """

    def test_default(self):
        service = api_service.APIService()

        @service.api(path='/test')
        def test(params: List[str]):  # pylint: disable=unused-argument
            return 'test'

        self.assertEqual(len(service.routers), 1)
        self.assertEqual(service.routers[0].func, test)
        self.assertEqual(service.routers[0].input_model, None)
        self.assertEqual(service.routers[0].output_model, None)
        self.assertEqual(service.routers[0].path, '/test')

    def test_multi(self):
        service = api_service.APIService()

        @service.api(path='/test1')
        def test1(params: List[str]):  # pylint: disable=unused-argument
            return 'test1'

        @service.api(path='/test2')
        def test2(params: List[str]):  # pylint: disable=unused-argument
            return 'test2'

        self.assertEqual(len(service.routers), 2)
        self.assertEqual(service.routers[0].func, test1)
        self.assertEqual(service.routers[0].input_model, None)
        self.assertEqual(service.routers[0].output_model, None)
        self.assertEqual(service.routers[0].path, '/test1')

        self.assertEqual(service.routers[1].func, test2)
        self.assertEqual(service.routers[1].input_model, None)
        self.assertEqual(service.routers[1].output_model, None)
        self.assertEqual(service.routers[1].path, '/test2')


class TestBuilder(unittest.TestCase):
    """
    Test towhee.api_service.build_service
    """

    def test_default(self):
        service = api_service.build_service([(lambda x: x, '/echo'), (lambda x: x + 1, '/add_one')])
        self.assertEqual(len(service.routers), 2)
        self.assertEqual(service.routers[0].input_model, None)
        self.assertEqual(service.routers[0].output_model, None)
        self.assertEqual(service.routers[0].path, '/echo')
        self.assertEqual(service.routers[1].input_model, None)
        self.assertEqual(service.routers[1].output_model, None)
        self.assertEqual(service.routers[1].path, '/add_one')
