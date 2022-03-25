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

try:
    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score # pylint: disable=unused-import
except ModuleNotFoundError as e:
    engine_log.error('sklearn not found, you can install via `pip install scikit-learn`.')
    raise ModuleNotFoundError('sklearn not found, you can install via `pip install scikit-learn`.') from e
