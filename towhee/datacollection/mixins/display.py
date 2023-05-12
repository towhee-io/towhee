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


class DisplayMixin: # pragma: no cover
    """
    Mixin for displaying data.
    """

    def prepare_table_data(self, limit=5):
        if limit > 0:
            data = [list(x.__dict__.values()) for i, x in enumerate(self) if i < limit]
        else:
            data = [list(x.__dict__.values()) for i, x in enumerate(self)]
        return {'data': data, 'headers': self._schema}

    def show(self, limit=5, tablefmt=None):
        """Print the first n lines of a DataCollection.

        Args:
            limit (int, optional): The number of lines to print. Prints all if limit is negative. Defaults to 5.
            tablefmt (str, optional): The format of the output, supports html, grid.
        """

        if not tablefmt:
            try:
                _ = get_ipython().__class__.__name__
                tablefmt = 'html'
            except NameError:
                tablefmt = 'grid'

        if tablefmt == 'html':
            from towhee.utils.html_table import NestedHTMLTable # pylint: disable=import-outside-toplevel
            NestedHTMLTable(self.prepare_table_data(limit)).show()
        else:
            from towhee.utils.console_table import NestedConsoleTable # pylint: disable=import-outside-toplevel
            NestedConsoleTable(self.prepare_table_data(limit)).show()

