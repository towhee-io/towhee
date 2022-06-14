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

class EntityView:
    """
    The view to iterate DataFrames.

    Args:
        offset (`int`):
            The offset of an Entity in the table.

    Examples:

    >>> from towhee import Entity, DataFrame
    >>> e = [Entity(a=a, b=b) for a,b in zip(range(3), range(3))]
    >>> df = DataFrame(e)
    >>> df = df.to_column()
    >>> df.to_list()[0]
    <EntityView dict_keys(['a', 'b'])>
    >>> df.to_list()[0].a
    0
    >>> df.to_list()[0].b
    0
    """

    def __init__(self, offset: int, table):
        self._offset = offset
        self._table = table

    def __getattr__(self, name):
        value = self._table[name][self._offset]
        try:
            return value.as_py()
        # pylint: disable=bare-except
        except:
            return value

    def __setattr__(self, name, value):
        if name in ('_table', '_offset'):
            self.__dict__[name] = value
            return
        self._table.write(name, self._offset, value)

        if self._offset == len(self._table) - 1:
            self._table.seal()
        self._table = self._table.prepare()

    def __repr__(self):
        """
        Define the representation of the EntityView.

        Examples:

        >>> from towhee import Entity, DataFrame
        >>> e = [Entity(a=a, b=b) for a,b in zip(range(5), range(5))]
        >>> df = DataFrame(e)
        >>> df = df.to_column()
        >>> df.to_list()[0]
        <EntityView dict_keys(['a', 'b'])>
        """
        return f'<{self.__class__.__name__} dict_keys({self._table.column_names})>'


if __name__ == '__main__':
    import doctest
    doctest.testmod()
