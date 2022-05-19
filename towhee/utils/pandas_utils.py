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

from towhee.utils.log import engine_log
import os

try:
    # pylint: disable=unused-import
    import pandas
except ModuleNotFoundError as moduleNotFound:
    engine_log.error('pandas not found, try to install pandas automatically.')

    try:
        # try to install automatically, and reload the `pandas` module
        os.system("pip install pandas")
        # pylint: disable=unused-import
        import pandas
    except:
        engine_log.error('pandas not found, you can install via `pip install pandas`.')
        raise ModuleNotFoundError('pandas not found, you can install via `pip install pandas`.') from moduleNotFound
