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


def intend(text, num_space=4):
    if isinstance(text, str):
        return ' ' * num_space + text
    elif isinstance(text, list):
        return [' ' * num_space + l for l in text]
    else:
        raise ValueError('unrecognized text format')


def add_line_separator(text):
    if isinstance(text, str):
        return text + '\n'
    elif isinstance(text, list):
        return [l + '\n' for l in text]
    else:
        raise ValueError('unrecognized text format')
