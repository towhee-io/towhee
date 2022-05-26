# Copyright 2022 Zilliz. All rights reserved.
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
import subprocess
import sys
import os


def prompt_install(package): # pragma: no cover
    """
    Function used to prompt user to install a package. If TOWHEE_WORKER env variable is set
    to True then the package will be automatically installed.
    """
    if os.getenv('TOWHEE_WORKER', 'False') == 'True' or get_yes_no(f'Do you want to install {package}?'):
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f'{package} installed successfully!')
        except subprocess.CalledProcessError:
            print(f'Ran into error installing {package}.')

def get_yes_no(question): # pragma: no cover
    while True:
        answer = input(question + ' [y/n]: ').lower()
        if 'yes'.startswith(answer.lower()):
            return True
        elif 'no'.startswith(answer.lower()):
            return False
        else:
            print('Not a valid answer.')
