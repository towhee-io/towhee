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


class HandlerMixin:
    """
    This mixin creates handler management interfaces
    Args:
        handler_prefix: (`str`)
            `HandlerMixin` will create a set of handler management interfaces based on
            each handler_prefix.
            For instance, given a `handler_prefix`='task_finish', there will be two
            methods that are generated, `add_task_finish_handler`
            and `call_task_finish_handlers`.
    """

    def add_handler_methods(self, *handler_prefix):

        class _HandlerMgr:
            """
            Each type of handlers are managed by one `_HandlerMgr`
            """

            def __init__(self):
                self._handlers = []

            def add_handler(self, handler):
                if isinstance(handler, list):
                    for h in handler:
                        if not callable(h):
                            raise AttributeError('handler must be callable')
                    self._handlers += handler
                elif callable(handler):
                    self._handlers.append(handler)
                else:
                    raise AttributeError('handler must be callable')

            def call_handlers(self, *args):
                for h in self._handlers:
                    h(*args)

            @property
            def handlers(self):
                return self._handlers

        self._handler_mgrs = []

        for prefix in handler_prefix:
            if isinstance(prefix, str):
                handler_mgr = _HandlerMgr()
                self._handler_mgrs.append(handler_mgr)

                add_handler_method_name = 'add_' + prefix + '_handler'
                self.__setattr__(add_handler_method_name, handler_mgr.add_handler)
                call_handler_method_name = 'call_' + prefix + '_handlers'
                self.__setattr__(call_handler_method_name, handler_mgr.call_handlers)
                handler_getter_name = prefix + '_handlers'
                self.__setattr__(handler_getter_name, handler_mgr.handlers)
            else:
                raise AttributeError(
                    '`HandlerMixin` can only take `str` as handler prefix, but found %s.'
                    % (type(prefix)))
