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
import os
from setuptools import setup, find_packages
from setuptools.command.install import install
import unittest
from pathlib import Path
import shutil


def test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('towhee/tests', pattern='test_*.py')
    return test_suite


def create_towhee_cache(src: str, dst: str, is_dir: bool = False):
    if not Path(dst).is_dir():
        Path(dst).mkdir(parents=True)
        if is_dir:
            dst = os.path.join(dst, src.split('/')[-1])
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)


class PostInstallCommand(install):
    def run(self):
        from towhee.engine import LOCAL_OPERATOR_CACHE, LOCAL_PIPELINE_CACHE, MOCK_OPS, MOCK_PIPELINES
        install.run(self)
        create_towhee_cache(MOCK_PIPELINES, LOCAL_PIPELINE_CACHE)
        create_towhee_cache(MOCK_OPS, LOCAL_OPERATOR_CACHE, True)


setup(
    name="towhee",
    version="0.1.3",
    description="",
    author="Towhee Team",
    author_email="towhee-team@zilliz.com",
    url="https://github.com/towhee-io/towhee",
    test_suite="setup.test_suite",
    cmdclass={'install': PostInstallCommand},
    install_requires=[
        'torch>=1.2.0',
        'torchvision>=0.4.0',
        'numpy>=1.19.5',
        'pandas>=1.1.5',
        'pyyaml>=5.3.0',
        'requests>=2.12.5',
        'tqdm>=4.59.0',
        'pillow>=8.3.1',
        'scipy>=1.5.3',
    ],
    packages=find_packages(),
    package_data={'towhee.tests.test_util': ['*.yaml']},
    license="http://www.apache.org/licenses/LICENSE-2.0",
)