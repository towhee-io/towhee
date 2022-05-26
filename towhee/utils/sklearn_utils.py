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
try:
    # pylint: disable=unused-import
    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, ConfusionMatrixDisplay
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import svm
except ModuleNotFoundError as moduleNotFound:
    try:
        from towhee.utils.dependency_control import prompt_install
        prompt_install('sklearn')
        # pylint: disable=unused-import
        from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, ConfusionMatrixDisplay
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn import svm
    except:
        from towhee.utils.log import engine_log
        engine_log.error('sklearn not found, you can install via `pip install sklearn`.')
        raise ModuleNotFoundError('sklearn not found, you can install via `pip install sklearn`.') from moduleNotFound
