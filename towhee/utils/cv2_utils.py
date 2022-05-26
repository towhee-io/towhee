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
    import cv2
except ModuleNotFoundError as e:
    try:
        from towhee.utils.dependency_control import prompt_install
        prompt_install('opencv-python')
        import cv2 # pylint: disable=ungrouped-imports
    except:
        from towhee.utils.log import engine_log
        engine_log.error('cv2 not found, you can install via `pip install opencv-python`.')
        raise ModuleNotFoundError('cv2 not found, you can install via `pip install opencv-python`.') from moduleNotFound
