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

from typing import Union
from pathlib import Path

from towhee.functional.entity import Entity


# pylint: disable=import-outside-toplevel
# pylint: disable=unnecessary-comprehension
class SaveMixin:
    """
    Mixin for saving data
    """
    def to_csv(self, csv_path: Union[str, Path], encoding: str = 'utf-8-sig'):
        """
        Save dc as a csv file.

        Args:
            csv_path (`Union[str, Path]`):
                The path to save the dc to.
            encoding (str):
                The encoding to use in the output file.
        """
        import csv
        from towhee.utils.pandas_utils import pandas as pd

        if isinstance(self._iterable, pd.DataFrame):
            self._iterable.to_csv(csv_path, index=False)
        else:
            with open(csv_path, 'w', encoding=encoding) as f:
                header = None
                writer = None

                def inner(row):
                    nonlocal header
                    nonlocal writer
                    if isinstance(row, Entity):
                        if not header:
                            header = row.__dict__.keys()
                            writer = csv.DictWriter(f, fieldnames=header)
                            writer.writeheader()
                        writer.writerow(row.__dict__)
                    else:
                        writer = writer if writer else csv.writer(f)
                        writer.writerow(row)

                any(map(inner, self._iterable))
