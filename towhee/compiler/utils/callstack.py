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
    def __init__(self, ignore: int = 0):
        """
        Initialize a Callstack.

        Args:
            ignore: the number of frames to ignore on the top of the callstack.

        Returns:
            a Callstack object ignoring a certain number of frames on th top.
        """
        self.frames = inspect.stack()
        # ignore the frame of Callstack.__init__
        ignore += 1
        del self.frames[0:ignore]
        self.stack_size = len(self.frames)
        return None

    def num_frames(self) -> int:
        """
        Get the number of frames.
        """
        return self.stack_size

    def find_func(self, func_name: str) -> int:
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
            found, return None. Suppose we have a callstack obejct `s`.
            When searching for an existing function `func_a`:
            >>> index_1 = s.find_func('func_a')
            >>> print(index_1)
            0 
            When searchiing for a function `func_b` does not exist:
            >>> index_2 = s.find_func('func_b')
            >>> print(index_2)
            None
        """
        for i in range(self.stack_size - 1, -1, -1):
            if self.frames[i].function == func_name:
                return i
        return None

    def hash(self, start: int = None, end: int = None, items: List[str] = None) -> str:
        """
        Get the hash value of the attributes contained in `items` between index `start`
        and `end` (includes `start`, excludes `end`).

        Args:
            start: the index of the start frame.
            end: the index of the end frame.
            items: the items to be hashed. Supported items are {filename, lineno,
                function, code_context, position, lasti}, where code_context denotes 
                the current line of code of the context, position denotes the frame's 
                index of the callstack, lasti denotes the index of last attempted 
                instruction in bytecode.

        Returns:
            The hash value.

        Raises:
            AttributeError: If an item in `items` is not supported, i.e. not one of
            {filename, lineno, function, code_context, position, lasti}.

        Examples:
            Suppose we have a callstack obejct `s` contains 2 frames.
            >>> hash_val = s.hash(0,1,['filename'])
            >>> print(hash_val)
            b284a28710cce90d9d9be3a7f4cabc8e 
            >>> hash_val = s.hash(0,5,['filename'])
            >>> print(hash_val)
            ERROR:root:index range [0, 4] out of list range [0, 1]
            None 
            >>> hash_val = s.hash(0,1,['attr_a'])
            >>> print(hash_val)
            Traceback (most recent call last):
              ...
            AttributeError: {'attr_a'} not supported
        """
        start = start or 0
        end = end or self.stack_size

        if end > self.stack_size or end <= 0 or start >= self.stack_size or start < 0 :
            logging.error(
                f"index range [{start}, {end}) out of list range" 
                f"[0, {self.stack_size})"
            )
            return None
        if start >= end:
            logging.error(f"end = {end} is less than or equal to start = {start}")
            return None

        full_item = {
            "filename", "lineno", "function", "code_context", "position", "lasti"
        }
        if not set(items).issubset(set(full_item)):
            invalid_item = set(items) - (set(items) & full_item)
            raise AttributeError(f"{invalid_item} not supported")

        md5 = hashlib.md5()
        for i, frame in enumerate(self.frames[start:end]):
            frame_dict = frame._asdict()
            frame_dict["position"] = i + start
            frame_dict["lasti"] = frame_dict["frame"].f_lasti
            frame_dict["code_context"] = "".join(frame_dict["code_context"])
            for item in items:
                md5.update(str(frame_dict[item]).encode("utf-8"))
        return md5.hexdigest()