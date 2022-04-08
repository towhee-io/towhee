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
from typing import Dict, Any, Optional, Set, Union, List

from towhee.functional.entity import Entity
from towhee.hparam import param_scope


def _select_callback(self):
    # pylint: disable=unused-argument
    # pylint: disable=protected-access
    def wrapper(_: str, index, *arg, **kws):
        if isinstance(index, str):
            index = (index,)
        if index is None and arg is not None and len(arg) > 0:
            index = arg

        def inner(entity: Entity):
            if index is not None:
                data = {column: getattr(entity, column) for column in index}
                return Entity(**data)
            return entity

        return self._factory(map(inner, self._iterable))
    return wrapper

class EntityMixin:
    """
    Mixin to help deal with Entity.

    Examples:

    1. define an operator with `register` decorator

    >>> from towhee import register
    >>> from towhee import DataCollection
    >>> @register
    ... def add_1(x):
    ...     return x+1

    2. apply the operator to named field of entity and save result to another named field

    >>> (
    ...     DataCollection([dict(a=1, b=2), dict(a=2, b=3)])
    ...         .as_entity()
    ...         .add_1['a', 'c']() # <-- use field `a` as input and filed `c` as output
    ...         .as_str()
    ...         .to_list()
    ... )
    ['{"a": 1, "b": 2, "c": 2}', '{"a": 2, "b": 3, "c": 3}']

    Select the entity on the specified fields.

    Examples:

    1. Select the entity on one specified field:

    >>> from towhee import Entity
    >>> from towhee import DataCollection
    >>> dc = DataCollection([Entity(a=i, b=i, c=i) for i in range(2)])
    >>> dc.select['a']().to_list()
    [<Entity dict_keys(['a'])>, <Entity dict_keys(['a'])>]

    2. Select multiple fields and unpack the entity:

    >>> (
    ...     DataCollection([Entity(a=i, b=i, c=i) for i in range(5)])
    ...         .select['a', 'b']()
    ...         .as_raw()
    ...         .to_list()
    ... )
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

    3. Another field selection syntax (not suggested):

    >>> (
    ...     DataCollection([Entity(a=i, b=i, c=i) for i in range(5)])
    ...         .select('a', 'b')
    ...         .as_raw()
    ...         .to_list()
    ... )
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    """
    def __init__(self):
        super().__init__()
        self.select = param_scope().callholder(_select_callback(self))

    # pylint: disable=invalid-name
    def fill_entity(self, _DefaultKVs: Optional[Dict[str, Any]] = None, _ReplaceNoneValue: bool = False, **kws):
        """
        When DataCollection's iterable exists of Entities and some indexes missing, fill default value for those indexes.

        Args:
            _ReplaceNoneValue (`bool`):
                Whether to replace None in Entity's value.
            _DefaultKVs (`Dict[str, Any]`):
                The key-value pairs stored in a dict.
        """
        if _DefaultKVs:
            kws.update(_DefaultKVs)

        def fill(entity: Entity):
            for k, v in kws.items():
                if not hasattr(entity, k):
                    setattr(entity, k, v)
                if _ReplaceNoneValue and v is None:
                    setattr(entity, k, 0)
            return entity

        return self._factory(map(fill, self._iterable))

    def as_entity(self, schema: Optional[List[str]]=None):
        """
        Convert elements into Entities.

        Args:
            schema (Optional[List[str]]):
                schema contains field names.

        Examples:
        1. convert dicts into entities:

        >>> from towhee import DataCollection
        >>> (
        ...     DataCollection([dict(a=1, b=2), dict(a=2, b=3)])
        ...         .as_entity()
        ...         .as_str()
        ...         .to_list()
        ... )
        ['{"a": 1, "b": 2}', '{"a": 2, "b": 3}']

        2. convert tuples into entities:

        >>> from towhee import DataCollection
        >>> (
        ...     DataCollection([(1, 2), (2, 3)])
        ...         .as_entity(schema=['a', 'b'])
        ...         .as_str()
        ...         .to_list()
        ... )
        ['{"a": 1, "b": 2}', '{"a": 2, "b": 3}']

        3. convert single value into entities:

        >>> from towhee import DataCollection
        >>> (
        ...     DataCollection([1, 2])
        ...         .as_entity(schema=['a'])
        ...         .as_str()
        ...         .to_list()
        ... )
        ['{"a": 1}', '{"a": 2}']
        """

        if schema is None:
            def inner(x):
                return Entity(**x)
        else:
            def inner(x):
                if len(schema) == 1:
                    x = (x, )
                data = dict(zip(schema, x))
                return Entity(**data)
        return self._factory(map(inner, self._iterable))

    def as_raw(self):
        """
        Convert entitis into raw python values

        Examples:

        1. unpack multiple values from entities:

        >>> from towhee import DataCollection
        >>> (
        ...     DataCollection([(1, 2), (2, 3)])
        ...         .as_entity(schema=['a', 'b'])
        ...         .as_raw()
        ...         .to_list()
        ... )
        [(1, 2), (2, 3)]

        2. unpack single value from entities:

        >>> (
        ...     DataCollection([1, 2])
        ...         .as_entity(schema=['a'])
        ...         .as_raw()
        ...         .to_list()
        ... )
        [1, 2]
        """

        def inner(x):
            if len(x.__dict__) == 1:
                return list(x.__dict__.values())[0]
            return tuple(getattr(x, name) for name in x.__dict__)
        return self._factory(map(inner, self._iterable))

    def replace(self, **kws):
        """
        Replace specific attributes with given vlues.
        """
        def inner(entity: Entity):
            for index, convert_dict in kws.items():
                origin_value = getattr(entity, index)
                if origin_value in convert_dict:
                    setattr(entity, index, convert_dict[origin_value])

            return entity

        return self._factory(map(inner, self._iterable))

    def dropna(self, na: Set[str] = {'', None}) -> Union[bool, 'DataCollection']:  # pylint: disable=dangerous-default-value
        """
        Drop entities that contain some specific values.

        Args:
            na (`Set[str]`):
                Those entities contain values in na will be dropped.
        """
        def inner(entity: Entity):
            for val in entity.__dict__.values():
                if val in na:
                    return False

            return True

        return self._factory(filter(inner, self._iterable))
    def rename(self, column: Dict[str, str]):
        """
        Rename an column in DataCollection.

        Args:
            column (`Dict[str, str]`):
                The columns to rename and their corresponding new name.
        """
        def inner(x):
            for key in column:
                x.__dict__[column[key]] = x.__dict__.pop(key)
            return x

        return self._factory(map(inner, self._iterable))


if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=False)
