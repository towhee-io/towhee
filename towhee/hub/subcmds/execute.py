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

from towhee.hparam import param_scope

class ExecuteCommand:
    """
    Implementation for subcmd `towhee execute`
    """
    def __init__(self, args) -> None:
        self._args = args

    def __call__(self):
        with param_scope(*self._args.define) as hp:
            import sys        # pylint: disable=import-outside-toplevel
            sys.path.append('.')
            import threading  # pylint: disable=import-outside-toplevel
            threading.setprofile(lambda f, e, a: param_scope.init(hp))

            if self._args.dry_run:
                hp().towhee.dry_run = True
            from towhee import pipeline  # pylint: disable=import-outside-toplevel
            pipe = pipeline(self._args.pipeline)

            if self._args.dry_run:
                print(repr(pipe))
            else:
                i = 0
                while i < self._args.iterations:
                    output = pipe(0)[0][0]

                    if self._args.output == 'imshow':
                        import cv2 # pylint: disable=import-outside-toplevel
                        cv2.imshow('imshow', output)
                        cv2.waitKey(1)
                    i += 1


    @staticmethod
    def install(subparsers):
        parser_execute = subparsers.add_parser('execute',
                                               help='execute pipeline')
        parser_execute.add_argument('-d',
                                    '--dry_run',
                                    action='store_true',
                                    help='dry run')
        parser_execute.add_argument('-n', '--iterations', type=int, default=-1)
        parser_execute.add_argument('-o',
                                    '--output',
                                    type=str,
                                    default='imshow')
        parser_execute.add_argument('-D', '--define', nargs='*', default=[])
        parser_execute.add_argument('pipeline', type=str, help='pipeline uri')
