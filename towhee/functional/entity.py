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
import os
from typing import overload, Dict, Any, Optional, Tuple


class Entity:
    """
    Entity is the basic data type in DataCollection.

    Users can create an Entity with free schema, which means there is no limit on attribute name and type.
    """
    @overload
    def __init__(self):
        """
        Create an empty entity.
        """
        ...

    @overload
    def __init__(self, target: Dict[str, Any]):
        """
        Create an Entity with the attibutes stored in a dict.
        """
        ...

    def __init__(self, target: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Create an Entity with given attributes.
        """
        self._data = ['id']
        self.__setattr__('id', os.urandom(16).hex())

        if target:
            kwargs.update(target)

        for k, v in kwargs.items():
            self.__setattr__(k, v)
            self._data.append(k)

    @property
    def info(self) -> Tuple[str]:
        """
        Return the attributes this eneity contains in the form of a tuple.
        """
        return tuple(self._data)

    def __repr__(self):
        """
        Define the representation of the Entity.
        """
        content = str(self.info)
        content += f' at {getattr(self, "id", id(self))}'
        return f'<{self.__class__.__name__} {content.strip()}>'

    def register(self, index: str):
        self._data.append(index)
