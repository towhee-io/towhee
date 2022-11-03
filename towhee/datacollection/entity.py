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

    def __getitem__(self, key):
        """
        Get the item from an entity.

        Examples:
            >>> from towhee.datacollection.entity import Entity
            >>> e = Entity(a=1, b=2)
            >>> e
            <Entity dict_keys(['a', 'b'])>
            >>> e['a']
            1
            >>> e['b']
            2
        """
        return getattr(self, key)

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
