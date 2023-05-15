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

from torch.utils.data.dataset import Dataset


class TowheeDataSet:
    """
    TowheeDataSet is a kind of dataset wrapper, where the `self.dataset` is the true dataset.
    """
    def __init__(self):
        self.framework = None
        self.dataset = None

    def __len__(self):
        raise NotImplementedError

class TorchDataSet(TowheeDataSet):
    """
    pytorch dataset
    """
    def __init__(self, torchDataSet: Dataset = None):
        super().__init__()
        self.framework = 'torch'
        self.dataset = torchDataSet

    def __len__(self):
        return len(self.dataset)

    def get_framework(self):
        return str(self.framework)
