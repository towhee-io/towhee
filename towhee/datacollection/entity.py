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
from typing import  Dict, Any

class Entity:
    """
    Entity is the basic data type in DataCollection.

    Users can create an Entity with free schema, which means there is no limit on
    attribute name and type.

    Exampels:
        >>> from towhee.datacollection.entity import Entity
        >>> e = Entity(a=1, b=2)
        >>> e
        <Entity dict_keys(['a', 'b'])>
        >>> e.a
        1
        >>> e.b
        2
    """
    def __init__(self, **kwargs):
        """
        Create an Entity with given attributes.
        """
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __repr__(self):
        """
        Define the representation of the Entity.

        Returns (`str`):
            The repr of the entity.

        Examples:
            >>> from towhee.datacollection.entity import Entity
            >>> e = Entity(a = 1, b = 2)
            >>> e
            <Entity dict_keys(['a', 'b'])>
        """
        content = repr(self.__dict__.keys())
        return f'<{self.__class__.__name__} {content.strip()}>'

    def __str__(self) -> str:
        """
        Return the str representation of an Entity.

        Returns (`str`):
            The string representation of the entity.

        Examples:
            >>> from towhee.datacollection.entity import Entity
            >>> e = Entity(a = 1, b = 2)
            >>> str(e)
            "{'a': 1, 'b': 2}"
        """
        return str(self.__dict__)

    def combine(self, *entities):
        """
        Combine entities.

        Args:
            entities (Entity): The entities to be added into self.

        Examples:
            >>> from towhee.datacollection.entity import Entity
            >>> e = Entity(a = 1, b = 2)
            >>> b = Entity(c = 3, d = 4)
            >>> c = Entity(e = 5, f = 6)
            >>> e.combine(b, c)
            >>> str(e)
            "{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}"
        """
        for x in entities:
            self.__dict__.update(x.__dict__)

    @classmethod
    def from_dict(cls, tar: Dict[str, Any]):
        """
        Create an Entity from a dict.

        Args:
            tar (`Dict[str, Any]`):
                The dict to create the Entity.

        Examples:
            >>> from towhee.datacollection.entity import Entity
            >>> d = {'a': 1, 'b': 2}
            >>> str(Entity.from_dict(d))
            "{'a': 1, 'b': 2}"
        """
        return cls(**tar)


class EntityView:
    """
    The view to iterate DataFrames.

    Args:
        offset (`int`):
            The offset of an Entity in the table.
        table:
            Which table stored in.

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
        """
        Get the value of the entity.

        Args:
            name (`str`):
                The key string of the entity.

        Returns:
            Any: The value for that entity at the current offset.
        """
        value = self._table[name][self._offset]
        try:
            return value.as_py()
        # pylint: disable=bare-except
        except:
            return value

    def __setattr__(self, name, value):
        """
        Set the `name` entity at current offset.

        Args:
            name (`str`):
                The entity name.
            value (`Any`):
                The entity value to be set.
        """
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

        Returns (`str`):
            The representation of the EntityView.

        Examples:

            >>> from towhee import Entity, DataFrame
            >>> e = [Entity(a=a, b=b) for a,b in zip(range(5), range(5))]
            >>> df = DataFrame(e)
            >>> df = df.to_column()
            >>> df.to_list()[0]
            <EntityView dict_keys(['a', 'b'])>

            >>> df = DataFrame(e)
            >>> df = df.set_chunksize(2)
            >>> df.to_list()[0]
            <EntityView dict_keys(['a', 'b'])>
        """
        return f'<{self.__class__.__name__} dict_keys({self._table.column_names})>'
