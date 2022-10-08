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
import copy

from towhee.functional import Entity
from towhee.functional.mixins import DisplayMixin
# pylint: disable=protected-access
class DataCollection(DisplayMixin):
    """
    A pythonic computation and processing framework.

    DataCollection is a pythonic computation and processing framework for unstructured
    data in machine learning and data science. It allows a data scientist or researcher
    to assemble data processing pipelines and do their model work (embedding,
    transforming, or classification) with a method-chaining style API. 
    """
    def __init__(self, data):
        self._data = data

    def __iter__(self):
        for _ in range(self._data.size):
            vals = self._data.get()
            keys = [k[0] for k in self._data.schema]
            yield Entity.from_dict(dict(zip(keys, vals)))

    # def __getitem__(self, index):
    #     pass

    # def __setitem__(self, index, value):
    #     pass

    def __add__(self, another: 'DataCollection') -> 'DataCollection':
        """
        Concat two DataCollections with same Schema.
        """
        for entity in another:
            self._data.put(list(entity.__dict__.values()))

        return self

    def __repr__(self) -> str:
        """
        String representation of the DataCollection.
        """
        names = self._data._schema.col_names
        types = self._data._schema.col_types
        content = ', '.join([i + ': ' + str(j) for i, j in zip(names, types)])
        return f'<{self.__class__.__name__} {content}>'

    def __len__(self):
        """
        Return the number of entities in the DataCollection.
        """
        return self._data.size

    def to_list(self) -> list:
        """
        Convert DataCollection to list.
        """
        return [entity for entity in self]

    # def show(self, limit=5, header=None, tablefmt='html', formatter={}):
    #     """
    #     Print the first n lines of a DataCollection.

    #     Args:
    #         limit (`int`):
    #             The number of lines to print. Prints all if limit is negative. Defaults to 5.
    #         header (`List[str]`):
    #             The field names. Defaults to None.
    #         tablefmt (`str`):
    #             The format of the output, supports 'html', 'plain', 'grid', etc. Defaults to 'html'.
    #     """
    #     contents = [x for i, x in enumerate(self) if i < limit]

    #     header = tuple(contents[0].__dict__) if not header else header
    #     data = [list(x.__dict__.values()) for x in contents]

    #     self.table_display(self.to_printable_table(data, header, tablefmt, formatter), tablefmt)

    def copy(self):
        """
        Copy a DataCollection.
        """
        return DataCollection(self._data.copy())

