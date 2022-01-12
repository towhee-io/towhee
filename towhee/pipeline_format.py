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
        it = out_df.map_iter()
        data = next(it)
        return data[0].value


class NormalFormat(OutputFormat):
    """
    Normal format.
    """

    def __call__(self, out_df: DataFrame):
        res = []
        it = out_df.map_iter()
        for data in it:
            # data is Tuple[Variable]
            data_value = []
            for item in data:
                if item is not None:
                    data_value.append(item.value)
            res.append(tuple(data_value))
        return res
