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


from setuptools import setup, find_packages

import unittest


def test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('towhee/tests', pattern='test_*.py')
    return test_suite


setup(
    name="towhee",
    version="0.1",
    description="",
    author="zilliz",
    author_email="",
    url="https://github.com/towhee-io/towhee",

    test_suite="setup.test_suite",

    install_requires=['torch>=1.2.0',
                      'torchvision>=0.4.0',
                      'pandas>=1.2.4',
                      'tqdm>=4.59.0'],

    packages=find_packages(),
    license="http://www.apache.org/licenses/LICENSE-2.0"
)
