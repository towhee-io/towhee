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

from abc import ABC, abstractmethod

from towhee.dataframe import DataFrame
from towhee.dataframe.iterators import MapIterator
from towhee.types._frame import _Frame


class OutputFormat(ABC):
    """
    Format pipeline's output.
    """

    @abstractmethod
    def __call__(self, out_df: DataFrame):
        pass

    @staticmethod
    def get_format_handler(pipeline_type: str):
        if pipeline_type == 'image-embedding':
            return ImageEmbeddingFormat()
        else:
            return NormalFormat()


class ImageEmbeddingFormat(OutputFormat):
    """
    Image embedding pipeline format.

    If pipelie's type is image-embedding, use the pipeline.
    """

    def __call__(self, out_df: DataFrame):
        it = MapIterator(out_df)
        data = next(it)
        return data[0][0]


class NormalFormat(OutputFormat):
    """
    Normal format.
    """

    def __call__(self, out_df: DataFrame):
        res = []
        it = MapIterator(out_df)
        for data in it:
            # data is Tuple
            data_value = []
            for item in data[0]:
                if item is not None and not isinstance(item, _Frame):
                    data_value.append(item)
            res.append(tuple(data_value))
        return res
