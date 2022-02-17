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

import numpy
from pathlib import Path
from typing import Union, Any

from towhee import pipeline


class ExecuteCommand:
    """
    Implementation for subcmd `towhee run`
    """
    def __init__(self, args) -> None:
        self._args = args

    def __call__(self):
        pipe = pipeline(self._args.pipeline)
        res = pipe(self._args.input)
        if not self._args.output:
            print(res)
        else:
            path = Path(self._args.output)
            self.save_result(path, res)

    @staticmethod
    def install(subparsers):
        parser_execute = subparsers.add_parser('run', help='execute command: run towhee pipeline')

        parser_execute.add_argument('-i', '--input', required=True, help='input the parameter for pipeline defaults to None')
        parser_execute.add_argument('-o', '--output', default=None, help='optional, path to the file that will be used to write results], '
                                                                         'defaults to None which will print the result')
        parser_execute.add_argument('pipeline', type=str, help='pipeline repo or path to yaml')

    @staticmethod
    def save_result(output: Union[str, Path], res: Any) -> None:
        """
        Save the results to local `output` file.

        Args:
            output (`str` | `Path`):
                The path that you are trying to save.
            res (`Any`):
                The result with any format.
        """
        file_name = Path(output) / 'towhee_output.txt'
        print(f'writing result to Path({str(output)})/towhee_output.txt')
        with open(str(file_name), 'w', encoding='utf-8') as f:
            if isinstance(res, list):
                f.write('[')
                for item in res:
                    f.write(f'{item}')
                f.write(']\n')
            elif isinstance(res, numpy.ndarray):
                numpy.savetxt(str(file_name), res)
            else:
                f.write(f'{str(res)}\n')
