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
import yaml
import requests
import os
from typing import List


class BaseRepr:
    """Base representation from which all other representation objects inherit.
    Primarily implements automatic serialization into YAML/YAML-like string formats,
    along with defining other universally used properties.

    Args:
        name:
            Name of the internal object described by this representation.
    """
    def __init__(self, name: str = None):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    def serialize(self) -> str:
        """Universal function used to serialize this representation.

        Returns:
            A string containing the serialized version of this representation. Example
            output:
                VariableRepr:
                    vtype: tensor
                    dtype: float'
        """
        raise NotImplementedError

    @staticmethod
    def load_str(string: str) -> List[dict]:
        """Load the representation(s) information from a YAML file (pre-loaded as string).

        Args:
            string(`str`):
                The string pre-loaded from a YAML.

        Returns:
            The list loaded from the YAML file that contains the representation(s) information.
        """
        res = yaml.safe_load(string)
        if isinstance(res, dict):
            res = [res]
        return res

    @staticmethod
    def load_file(file: str) -> List[dict]:
        """Load the representation(s) information from a local YAML file.

        Args:
            file(`str`):
                The file path.

        Returns:
            The list loaded from the YAML file that contains the representation(s) information.
        """
        with open(file, 'r', encoding='utf-8') as f:
            return BaseRepr.load_str(f)

    @staticmethod
    def load_url(url: str) -> List[dict]:
        """Load the representation(s) information from a remote YAML file.

        Args:
            url(`str`):
                The url points to the remote YAML file.

        Returns:
            The list loaded from the YAML file that contains the representation(s) information.
        """
        src = requests.get(url, timeout=5).text
        return BaseRepr.load_str(src)

    @staticmethod
    def load_src(file_or_src: str) -> List[dict]:
        """Load the information for the representation.

        We support file from local file/HTTP/HDFS.

        Args:
            file_or_src(`str`):
                The source YAML file or the URL points to the source file or a str
                loaded from source file.

        returns:
            The YAML file loaded as list.
        """
        # If `file_or_src` is a loacl file
        if os.path.isfile(file_or_src):
            return BaseRepr.load_file(file_or_src)
        # If `file_or_src` from HTTP
        elif file_or_src.lower().startswith('http'):
            return BaseRepr.load_url(file_or_src)
        # If `file_or_src` is neither a file nor url
        return BaseRepr.load_str(file_or_src)
