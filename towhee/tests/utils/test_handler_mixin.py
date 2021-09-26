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


import unittest

from towhee.utils import (
    HandlerMixin,
)


class TestHandlerMixin(unittest.TestCase):
    """Basic test case for `HandlerMixin`.
    """

    class CallbackAdd:
        def __init__(self):
            self.y = 0

        def add(self, x):
            self.y += x

    class CallbackSub:
        def __init__(self):
            self.y = 0

        def sub(self, x):
            self.y -= x

    class Foo(HandlerMixin):
        def __init__(self, x):
            self.add_handler_methods('cal')
            self._x = x

        def call_handlers(self):
            self.call_cal_handlers(self._x)

    def test_handler_add_and_call(self):

        foo = self.Foo(5)

        callback_add = self.CallbackAdd()
        callback_sub = self.CallbackSub()

        foo.add_cal_handler(callback_add.add)
        foo.add_cal_handler(callback_sub.sub)

        foo.call_handlers()

        self.assertEqual(callback_add.y, 5)
        self.assertEqual(callback_sub.y, -5)
