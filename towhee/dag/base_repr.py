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
import requests
import os
import logging
import json
from typing import Dict, Set, Any

from towhee.hparam import param_scope, HyperParameter
from towhee.utils.yaml_utils import dump_yaml, load_yaml


class BaseRepr:
    """
    Base representation from which all other representation objects inherit.

    Primarily implements automatic serialization into YAML/YAML-like string formats,
    along with defining other universally used properties.

    Args:
        name (`str`):
            Name of the internal object described by this representation.
    """
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    @staticmethod
    def is_valid(info: Dict[str, Any], essentials: Set[str]) -> bool:
        """
        Check if the src is a valid YAML file to describe a component in Towhee.

        Args:
            info (`Dict[str, Any]`):
                The dict loaded from the source file.
            essentials (`Set[str]`):
                The essential keys that a valid YAML file should contain.

        Returns:
            (`bool`)
                Return `True` if the src file is a valid YAML file to describe a
                component in Towhee, else `False`.
        """
        info_keys = set(info.keys())
        if not isinstance(info, dict) or not essentials.issubset(info_keys):
            logging.error('Info [%s] is not valid, lack attr [%s]', str(info), essentials - info_keys)
            return False
        return True

    @staticmethod
    def render_template(string: str):
        string = string.read() if hasattr(string, 'read') else string
        retval = load_yaml(string)
        retval = param_scope(**retval)
        variables = retval().variables({})
        with param_scope(**variables) as hp:
            variables.update(hp().variables({}))
            rendered = string.format(**variables)
        return rendered

    @staticmethod
    def inject_template(info: Dict[str, Any]) -> Dict:
        def inject(op, injections):
            if op['name'] in injections:
                patch = injections[op['name']]
                op = HyperParameter(**op)
                op.update(patch)
            return op

        with param_scope() as hp:
            if hp().injections(None) is not None:
                info['operators'] = [inject(op, hp().injections()) for op in info['operators']]
                if 'ir' in info:
                    del info['ir']
                info = json.loads(json.dumps(info))
                info['ir'] = dump_yaml(info)
        return info

    @staticmethod
    def load_str(string: str) -> dict:
        """
        Load the representation(s) information from a YAML file (pre-loaded as string).

        Args:
            string (`str`):
                The string pre-loaded from a YAML.

        Returns:
            (`dict`)
                The dict loaded from the YAML file that contains the representation
                information.
        """
        rendered = BaseRepr.render_template(string)
        info = load_yaml(rendered)
        info['ir'] = rendered
        info = BaseRepr.inject_template(info)
        return info

    @staticmethod
    def load_file(file: str) -> dict:
        """
        Load the representation(s) information from a local YAML file.

        Args:
            file (`str`):
                The file path.

        Returns:
            (`dict`)
                The dict loaded from the YAML file that contains the representation
                information.
        """
        with open(file, 'r', encoding='utf-8') as f:
            return BaseRepr.load_str(f)

    @staticmethod
    def load_url(url: str) -> dict:
        """
        Load the representation information from a remote YAML file.

        Args:
            url (`str`):
                The url points to the remote YAML file.

        Returns:
            (`dict`)
                The dict loaded from the YAML file that contains the representation
                information.
        """
        src = requests.get(url, timeout=5).text
        return BaseRepr.load_str(src)

    @staticmethod
    def load_src(file_or_src: str) -> dict:
        """
        Load the information for the representation. We support file from local
        file/HTTP/HDFS.

        Args:
            file_or_src (`str`):
                The source YAML file or the URL points to the source file or a str
                loaded from source file.

        returns:
            (`dict`)
                The YAML file loaded as dict.
        """
        # If `file_or_src` is a loacl file.
        if os.path.isfile(file_or_src):
            return BaseRepr.load_file(file_or_src)
        # If `file_or_src` from HTTP.
        elif file_or_src.lower().startswith('http'):
            return BaseRepr.load_url(file_or_src)
        # If `file_or_src` is neither a file nor url.
        return BaseRepr.load_str(file_or_src)
