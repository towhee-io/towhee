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
import argparse

from towhee.command.develop import DevelopRepo
from towhee.command.execute import ExecutePipeline


def main_body(args):
    parser = argparse.ArgumentParser('towhee', 'towhee')
    subparsers = parser.add_subparsers(dest='action', help='towhee command line tool.')

    actions = {
        'install': DevelopRepo,
        'develop': DevelopRepo,
        'run': ExecutePipeline
    }

    for _, impl in actions.items():
        impl.install(subparsers)

    parsed = parser.parse_args(args)
    if parsed.action is None:
        parser.print_help()
    else:
        actions[parsed.action](parsed)()


def main():
    main_body(sys.argv[1:])


if __name__ == '__main__':
    main()
