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


class DatasetMixin:
    """
        Mixin for dealing with dataset
    """

    # pylint: disable=import-outside-toplevel
    def image_decode(self):
        pass

    def image_encode(self):
        pass

    def audio_decode(self):
        pass

    def audio_encode(self):
        pass

    def video_decode(self):
        pass

    def video_encode(self):
        pass

    def from_json(self):
        pass

    def from_csv(self):
        pass

    def split_train_test(self):
        pass

    def save_json(self):
        pass

    def save_image(self):
        pass

    def save_csv(self):
        pass

    def get_info(self):
        pass
