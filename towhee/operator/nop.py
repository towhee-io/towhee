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


from typing import Any, Dict, NamedTuple

from towhee.operator import PyOperator


class NOPOperator(PyOperator):
    """No-op operator. Input arguments are redefined as a `NamedTuple` and returned as
    outputs.
    """

    def __init__(self):
        #pylint: disable=useless-super-delegation
        super().__init__()

    def __call__(self, **args: Dict[str, Any]) -> NamedTuple:
        fields = [(name, type(val)) for name, val in args.items()]
        return NamedTuple('Outputs', fields)(**args)  # pylint: disable=not-callable
