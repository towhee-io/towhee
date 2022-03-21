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
from typing import Dict, Any, Optional

from towhee.functional.entity import Entity


class EntityMixin:
    """
    Mixin to help deal with Entity.

    Examples:

    1. define an operator with `register` decorator

    >>> from towhee import register
    >>> from towhee.functional import DataCollection
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
    """

    def fill_entity(self, default_kvs: Optional[Dict[str, Any]] = None, **kws):
        """
        When DataCollection's iterable exists of Entities and some indexes missing, fill default value for those indexes.

        Args:
            default_kvs (`Dict[str, Any]`):
                The key-value pairs stored in a dict.
        """
        if default_kvs:
            kws.update(default_kvs)

        def fill(entity: Entity):
            for k, v in kws.items():
                if not hasattr(entity, k):
                    setattr(entity, k, v)
                    entity.register(k)
            return entity

        return self.factory(map(fill, self._iterable))

    def as_entity(self):
        """
        Convert elements into Entities.

        Examples:

        >>> from towhee.functional import DataCollection
        >>> (
        ...     DataCollection([dict(a=1, b=2), dict(a=2, b=3)])
        ...         .as_entity()
        ...         .as_str()
        ...         .to_list()
        ... )
        ['{"a": 1, "b": 2}', '{"a": 2, "b": 3}']
        """

        def inner(x):
            return Entity(**x)

        return self.factory(map(inner, self._iterable))


if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=False)
