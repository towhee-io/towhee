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
from typing import Iterable

from tabulate import tabulate


class DisplayMixin:
    """
        Mixin for display data
    """
    def as_str(self):
        return self.factory(map(str, self._iterable))

    def plot(self):
        pass

    def head(self, num: int = 5, header: Iterable[str] = None, numalign: str = 'center', stralign: str = 'center', tablefmt: str = 'plain'):
        """
        Print the first n lines in a DataCollection.

        Args:
            num (`int`):
                The number of lines to print. Default value is 5.
            header (`Iterable[str]`):
                The name of columns defined by users.
            numalign (`str`):
                How the nums align, support center, right, left.
            stralign (`str`):
                How the strs align, support center, right, left.
            tablefmt (`str`):
                The format of the output, support plain, simple, github, grid, fancy_grid, pipe, orgtbl, jira, presto, psql, rst, mediawiki,
                moinmoin, youtrack, html, latex, latex_raw, latex_booktabs, textile.
        """
        to_display = []

        cnt = 0
        for i in self._iterable:
            header = i.info[1:] if not header else header
            to_display.append(i.to_dict().values())
            cnt += 1
            if cnt == num:
                break

        print(tabulate(to_display, headers=header, tablefmt=tablefmt, numalign=numalign, stralign=stralign))

    def tail(self, num: int = 5, header: Iterable[str] = None, numalign: str = 'center', stralign: str = 'center', tablefmt: str = 'plain'):
        """
        Print the last n lines in a DataCollection.

        Args:
            num (`int`):
                The number of lines to print. Default value is 5.
            header (`Iterable[str]`):
                The name of columns defined by users.
            numalign (`str`):
                How the nums align, support center, right, left.
            stralign (`str`):
                How the strs align, support center, right, left.
            tablefmt (`str`):
                The format of the output, support plain, simple, github, grid, fancy_grid, pipe, orgtbl, jira, presto, psql, rst, mediawiki,
                moinmoin, youtrack, html, latex, latex_raw, latex_booktabs, textile.
        """
        to_display = []

        if self.is_stream:
            raise AttributeError('The DataCollection is stream, tail function not supported.')

        maxsize = len(self._iterable)
        for i in self._iterable[maxsize - num:]:
            header = i.info[1:] if not header else header
            to_display.append(i.to_dict().values())

        print(tabulate(to_display, headers=header, tablefmt=tablefmt, numalign=numalign, stralign=stralign))
