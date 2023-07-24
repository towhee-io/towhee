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
from typing import Optional, Any

from pydantic import BaseModel, validator
from towhee.utils.singleton import singleton


@singleton
class LogConfig(BaseModel):
    """
    Towhee logger config.

    Args:
        mode (`Optional[str]`):
            The mode of Logger, decide which Handler to use, 'file' or 'console', defaults to 'console'.
        rotate_mode (`optional[str]`):
            The mode of rotating files, 'time' or 'size', defaults to 'time'.
        filename (`Optional[str]`):
            Path for log files, defaults to 'towhee.log'.
        rotate_when (`Optional[str]`):
            The type of TimedRotatingFileHandler interval, supports 'S', 'M', 'H', 'D', 'W0'-'W6', 'midnight', deafults to 'D'.
        rotate_interval (`Optional[int]`):
            The interval value of timed rotating, defaults to 1.
        backup_count (`Optional[int]`):
            The number of log files to keep, defaults to 10.
        stream_level (`Optional[str]`):
            The log level in console mode, defaults to 'WARNING'.
        file_level (`Optional[str]`):
            The log level in file mode, defautls to 'INFO'.
        utc (`Optional[bool]`):
            Whether to use utc time, defaults to False.
        stream (`Optional[Any]`):
            The stream to write logs into in console mode, defeaults to None and sys.stderr is used.
    """
    mode: Optional[str] = 'console'
    rotate_mode: Optional[str] = 'time'
    filename: Optional[str] = 'towhee.log'
    file_max_size: Optional[int] = 100 * 1000 * 1000
    rotate_when: Optional[str] = 'D'
    rotate_interval: Optional[int] = 1
    backup_count: Optional[int] = 10
    stream_level: Optional[str] = 'WARNING'
    file_level: Optional[str] = 'INFO'
    utc: Optional[bool] = False
    stream: Optional[Any] = None

    @validator('mode')
    def mode_range(cls, mode):  # pylint: disable=no-self-argument
        if mode not in ['console', 'file']:
            raise ValueError(f'value should be either \'console\' or \'file\', not \'{mode}\'.')
        return mode

    @validator('rotate_mode')
    def rotate_mode_range(cls, rotate_mode):  # pylint: disable=no-self-argument
        if rotate_mode not in ['time', 'size']:
            raise ValueError(f'value should be either \'time\' or \'size\', not \'{rotate_mode}\'.')
        return rotate_mode
