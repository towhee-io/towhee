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


class Callstack:

    def collect(self):
        """
        Collect the frames of current callstack.
        """
        raise NotImplementedError

    def num_frames(self) -> int:
        """
        Get the number of frames
        """
        raise NotImplementedError

    def find_func(self, func_name: str) -> int:
        """
        Given a function name, find the first-match from the outermost frame

        Args:
            func_name: the function name to find.
        Returns:
            The first-matched frame index.

        Raises:

        Examples:
        """
        raise NotImplementedError

    def hash(self, start: int = None, end: int = None, items: list[str] = None) -> str:
        """
        Get a hash value from frames.

        Args:
            start: the index of the start frame.
            end: the index of the end frame.
            items: the items to be hashed. Supported items are {filename, lineno, 
                funcname, codectx, index, lasti}, where codectx denotes the current
                line of code of the context, index denotes the frame's index of the
                callstack, lasti denotes the index of last attempted instruction in 
                bytecode.

        Returns:
            The hash value.

        Raises:

        Examples:
        """
        raise NotImplementedError
