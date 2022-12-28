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

import numpy as np
from towhee.runtime.data_queue import Empty
from towhee.utils.np_format import NumpyFormat
from towhee.utils.empty_format import EmptyFormat


class TritonSerializer(json.JSONEncoder):
    """
    Support data that might occur in a towhee pipeline to json.
    """
    def default(self, o):
        if isinstance(o, np.ndarray):
            return NumpyFormat(o).to_dict()
        if o is Empty():
            return EmptyFormat().to_dict()

        return json.JSONEncoder.default(self, o)


class TritonParser(json.JSONDecoder):
    """
    Support data that might occur in a towhee pipeline from json.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):  # pylint: disable=method-hidden
        if '_NP' in dct:
            return NumpyFormat.from_dict(dct)
        if '_EMP' in dct:
            return EmptyFormat.from_dict()
        return dct


def to_triton_data(data):
    return json.dumps(data, cls=TritonSerializer)


def from_triton_data(data):
    return json.loads(data, cls=TritonParser)
