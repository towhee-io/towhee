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
from typing import List, Union
import hashlib


class Callstack:
    """
    A Callstack object contains the working frames at the moment.

    Args:
        ignore (`int`):
            The number of frames to ignore on the top of the callstack.
    """
    def __init__(self, ignore: int = 0):
        self.frames = inspect.stack()
        # ignore the frame of Callstack.__init__
        ignore += 1
        if ignore > len(self.frames):
            raise ValueError(f"ignore = {ignore-1} is out of frame range")
        del self.frames[0:ignore]
        self.size = len(self.frames)

    def num_frames(self) -> int:
        """
        Get the number of frames.

        Returns:
            (`int`)
                The size of current stack.
        """
        return self.size

    def find_func(self, func_name: str) -> Union[int, None]:
        """
        Given a function name, find the first-matched and outermost frame from current
        stack.

        Args:
            func_name (`str`):
                The function name to find.

        Returns:
            (`Union[int, None]`)
                If at least one matching frame exits, return the first-matched frame
                index. Else, return None.
        """
        for i in range(self.size - 1, -1, -1):
            if self.frames[i].function == func_name:
                return i
        return None

    def hash(self, start: int = None, end: int = None, items: List[str] = None) -> str:
        """
        Get the hash value of the attributes contained in `items` between index `start`
        and `end` (includes `start`, excludes `end`).

        Args:
            start (`int`):
                The index of the start frame.
            end (`int`):
                The index of the end frame.
            items (`List[str]`):
                The items to be hashed. Supported items are
                {filename, lineno, function, code_context, position, lasti}, where
                code_context denotes the current line of code of the context, position
                denotes the frame's index of the callstack, lasti denotes the index of
                last attempted instruction in bytecode.

        Returns:
            (`str`)
                The hash value.

        Raises:
            （`IndexError`）
                If the args [`start`, `end`) is out of the frame range or `end` less
                than `start`.
            （`ValueError`）
                If an item in `items` is not supported, i.e. not one of
                {filename, lineno, function, code_context, position, lasti}.
        """
        start = start or 0
        end = end or self.size

        if end > self.size or end <= 0 or start >= self.size or start < 0:
            raise IndexError(f"index range [{start}, {end}) out of frame range" f"[0, {self.size})")
        if start >= end:
            raise IndexError(f"end = {end} is less than or equal to start = {start}")

        full_item = {"filename", "lineno", "function", "code_context", "position", "lasti"}
        if not set(items).issubset(set(full_item)):
            invalid_item = set(items) - (set(items) & full_item)
            raise ValueError(f"{invalid_item} not supported")

        md5 = hashlib.md5()
        for i, frame in enumerate(self.frames[start:end]):
            frame_dict = frame._asdict()
            frame_dict["position"] = i + start
            frame_dict["lasti"] = frame_dict["frame"].f_lasti
            frame_dict["code_context"] = ("".join(frame_dict["code_context"]) if frame_dict["code_context"] else "")
            for item in items:
                md5.update(str(frame_dict[item]).encode("utf-8"))
        return md5.hexdigest()
