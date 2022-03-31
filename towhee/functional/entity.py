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
import json
from typing import overload


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

    def __init__(self, **kwargs):
        """
        Create an Entity with given attributes.
        """

        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __repr__(self):
        """
        Define the representation of the Entity.
        """
        content = str(self.__dict__.keys())
        return f'<{self.__class__.__name__} {content.strip()}>'

    def __str__(self) -> str:
        return json.dumps({k: getattr(self, k) for k in self.__dict__})

    def to_dict(self):
        return self.__dict__
