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
import inspect
from typing import List
import hashlib
import logging

class Callstack:

    @staticmethod
    def collect():
        """
        Collect the frames of current callstack.
        """
        frames = inspect.stack()
        return [f.frame for f in frames]

    @staticmethod
    def num_frames() -> int:
        """
        Get the number of frames
        """
        frames = inspect.stack()
        return len(frames)

    @staticmethod
    def find_func(func_name: str) -> int:
        """
        Given a function name, find the first-matched and outermost frame from current
        stack.

        Args:
            func_name: the function name to find.
        Returns:
            If at least one matching frame exits, return the first-matched frame index.
            else, return None.

        Examples:
            Callstack class provides a function to find the first-match frame in
            current stack to a given function name, and return its index. If not
            found, return None.
            >>> s = Callstack()
            >>> index_1 = s.find_func('find_func')
            >>> print(index_1)
            0
            >>> s = Callstack()
            >>> index_2 = s.find_func('collect')
            >>> print(index_2)
            None
        """
        frames = inspect.stack()
        for i in range(len(frames) - 1, -1, -1):
            if frames[i].function == func_name:
                return i
        return None

    @staticmethod
    def hash(start: int = None, end: int = None, items: List[str] = None) -> str:
        """
        Get the hash value of the attributes contained in `items` between index `start`
        and `end` (includes `start`, excludes `end`).

        Args:
            start: the index of the start frame.
            end: the index of the end frame.
            items: the items to be hashed. Supported items are {filename, lineno,
                funcname, code_context, index, lasti}, where codectx denotes the current
                line of code of the context, index denotes the frame's index of the
                callstack, lasti denotes the index of last attempted instruction in
                bytecode.

        Returns:
            The hash value.

        Raises:
            IndexError: If `end` or `start` out of `items` range or `end` is in front
            of `end`.
            AttributeError: If an item in `items` is not supported, i.e. not one of
            {filename, lineno, funcname, code_context, index, lasti}.

        Examples:
            >>> s = Callstack()
            >>> hash_val = s.hash(0,1,['filename'])
            >>> print(hash_val)
            b284a28710cce90d9d9be3a7f4cabc8e
            >>> s = Callstack()
            >>> hash_val = s.hash(0,5,['filename'])
            >>> print(hash_val)
            ERROR:root:index range [0, 4] out of list range [0, 1]
            None
            >>> s = Callstack()
            >>> hash_val = s.hash(0,5,['attr_a'])
            >>> print(hash_val)
            Traceback (most recent call last):
              ...
            AttributeError: {'attr_a'} not supported
        """
        frames = inspect.stack()

        start = len(frames) + start if start < 0 else start
        end = len(frames) + end if end < 0 else end
        if end > len(frames) or start >= len(frames) :
            logging.error(f"index range [{start}, {end - 1}] out of list range [0, {len(frames) - 1}]")
            return None
        if start >= end:
            logging.error(f"end = {end} is less than or equal to start = {start}")
            return None

        full_item = {"filename", "lineno", "funcname", "code_context", "index", "lasti"}
        if not set(items).issubset(set(full_item)):
            invalid_item = set(items) - (set(items) & full_item)
            raise AttributeError(f"{invalid_item} not supported")

        md5 = hashlib.md5()
        for i, frame in enumerate(frames[start:end]):
            for item in items:
                if item == "index":
                    md5.update(str(start + i).encode('utf-8'))
                elif item == "lasti":
                    md5.update(frame.frame.f_lasti.encode('utf-8'))
                else:
                    md5.update(getattr(frame, item).encode('utf-8'))

        return md5.hexdigest()