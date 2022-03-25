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

import sys
import unittest
from pathlib import Path
from typing import List

from setuptools import find_packages, setup
from setuptools.command.install import install


def test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('towhee/tests', pattern='test_*.py')
    return test_suite


def parse_requirements(file_name: str) -> List[str]:
    with open(file_name) as f:
        return [
            require.strip() for require in f
            if require.strip() and not require.startswith('#')
        ]


kws = {}
try:
    from sphinx.setup_command import \
        BuildDoc  # pylint: disable=import-outside-toplevel
    cmdclass = {'build_sphinx': BuildDoc}
    command_options = {
        'build_sphinx': {
            'project': ('setup.py', 'towhee'),
            'version': ('setup.py', '0.5.1'),
            'release': ('setup.py', '0.5.1'),
            'source_dir': ('setup.py', 'towhee/doc/source')
        }
    }
    kws = {'cmdclass': cmdclass, 'command_options': command_options}

except:  # pylint: disable=bare-except
    pass

if '--models' in sys.argv:
    sys.argv.remove('--models')
    setup(name='towhee.models',
          version='0.5.1',
          description='',
          author='Towhee Team',
          author_email='towhee-team@zilliz.com',
          use_scm_version={'local_scheme': 'no-local-version'},
          setup_requires=['setuptools_scm'],
          url='https://github.com/towhee-io/towhee',
          test_suite='setup.test_suite',
          install_requires=parse_requirements('requirements.txt'),
          extras_require={':python_version<\'3.7\'': 'importlib-resources'},
          tests_require=parse_requirements('test_requirements.txt'),
          packages=find_packages(include=['towhee.models*']),#['towhee.models'],
          package_data={'towhee.tests.test_util': ['*.yaml']},
          license='http://www.apache.org/licenses/LICENSE-2.0',
          entry_points={
              'console_scripts': ['towhee=towhee.command.cmdline:main'],
          },
          **kws)
else:
    setup(name='towhee',
          version='0.5.1',
          description='',
          author='Towhee Team',
          author_email='towhee-team@zilliz.com',
          use_scm_version={'local_scheme': 'no-local-version'},
          setup_requires=['setuptools_scm'],
          url='https://github.com/towhee-io/towhee',
          test_suite='setup.test_suite',
          install_requires=parse_requirements('requirements.txt'),
          extras_require={':python_version<\'3.7\'': 'importlib-resources'},
          tests_require=parse_requirements('test_requirements.txt'),
          packages=find_packages(exclude=['*test*', 'towhee.models*']),
          package_data={'towhee.tests.test_util': ['*.yaml']},
          license='http://www.apache.org/licenses/LICENSE-2.0',
          entry_points={
              'console_scripts': ['towhee=towhee.command.cmdline:main'],
          },
          **kws)
