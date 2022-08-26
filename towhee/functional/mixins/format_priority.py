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

from typing import List

from towhee.hparam.hyperparameter import param_scope


class FormatPriorityMixin:
    """Mixin for Format Priority.

    Examples:

        >>> import towhee
        >>> dc = towhee.DataCollection.range(100)
        >>> dc = dc.set_format_priority(['tensorrt', 'onnx'])
        >>> dc.get_formate_priority()
        ['tensorrt', 'onnx']
    """
    def __init__(self) -> None:
        super().__init__()
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        if parent is not None and hasattr(parent, '_format_priority'):
            self._format_priority = parent._format_priority

    def set_format_priority(self, format_priority: List[str]):
        """Set format priority.

        Args:
            format_priority (List[str]): The priority queue of format.

        Returns:
            DataCollection: DataCollection with format_priorty set.
        """
        self._format_priority = format_priority
        return self._factory(self._iterable)

    def get_formate_priority(self):
        return self._format_priority
