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
from typing import Iterable, Tuple
from tabulate import tabulate

from towhee.hparam import param_scope


def _value_counts_callback(self):
    def wrapper(_: str, index: Tuple[str], *arg, **kws):
        # pylint: disable=import-outside-toplevel
        # pylint: disable=unused-argument
        import pandas as pd
        columns = [[] for _ in index]
        for x in self:
            for i in range(len(index)):
                columns[i].append(getattr(x, index[i]))

        data = {}
        for i in range(len(index)):
            data[index[i]] = columns[i]
        df = pd.DataFrame(data)
        print(df.value_counts(**kws))

    return wrapper


def _plot_callback(self):
    def wrapper(_: str, index: Tuple[str], *arg, **kws):
        # pylint: disable=import-outside-toplevel
        # pylint: disable=unused-argument
        import pandas as pd
        columns = [[] for _ in index]
        for x in self:
            for i in range(len(index)):
                columns[i].append(getattr(x, index[i]))

        data = {}
        for i in range(len(index)):
            data[index[i]] = columns[i]
        df = pd.DataFrame(data)
        if 'kind' not in kws:
            kws.update(kind='hist')
        df.plot(**kws)

    return wrapper


class DisplayMixin:
    """
    Mixin for display data

    Example:

    >>> from towhee import DataCollection
    >>> from towhee import Entity
    >>> dc = DataCollection([Entity(a=i, b=i) for i in range(2)])
    >>> dc.value_counts['a'](normalize=True)
    a
    0    0.5
    1    0.5
    dtype: float64
    >>> dc.plot['a']()
    """
    def __init__(self):
        self.value_counts = param_scope().callholder(_value_counts_callback(self))
        self.plot = param_scope().callholder(_plot_callback(self))

    def as_str(self):
        return self.factory(map(str, self._iterable))

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
            header = i.to_dict().keys() if not header else header
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
            header = i.to_dict().keys() if not header else header
            to_display.append(i.to_dict().values())

        print(tabulate(to_display, headers=header, tablefmt=tablefmt, numalign=numalign, stralign=stralign))


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)
