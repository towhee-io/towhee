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
import io
from typing import Union, Dict, List

try:
    # pylint: disable=unused-import,ungrouped-imports
    import ruamel.yaml
    from ruamel.yaml import YAML
except ModuleNotFoundError as moduleNotFound:
    try:
        from towhee.utils.dependency_control import prompt_install
        prompt_install('ruamel.yaml')
        # pylint: disable=unused-import,ungrouped-imports
        import ruamel.yaml
        from ruamel.yaml import YAML
    except:
        from towhee.utils.log import engine_log
        engine_log.error('ruamel.yaml not found, you can install via `pip install ruamel.yaml`.')
        raise ModuleNotFoundError('ruamel.yaml not found, you can install via `pip install ruamel.yaml`.') from moduleNotFound


def load_yaml(stream, typ: str = 'safe'):
    """
    Load the YAML document in a stream as Python object.

    Args:
        stream:
            The YAML file loaded in a stream.
        typ:
            The type of Dumper/Loader to apply in ruamel.yaml.

    Returns:
        The YAML laoded as python object.
    """
    yaml = YAML(typ=typ)
    data = yaml.load(stream=stream)
    if not isinstance(data, dict):
        raise ValueError(
            'The loaded data should be a dict, please check your yaml source.\
            (Such error is very likely caused by using dash instead of underline in yaml file name.)'
        )

    return data


def dump_yaml(data: Union[str, Dict, List], stream=None):
    """
    Dump YAML file as python object.

    Args:
        data (`Union[str, Dict, List]`):
            The data to write into a YAMl file.
        stream:
            The stream to dump the YAML into.
    """
    yaml = YAML()
    yaml.indent(mapping=4, sequence=8, offset=4)
    yaml.compact_seq_map = False

    if stream is None:
        stream = io.StringIO()
        yaml.dump(data=data, stream=stream)
        return stream.getvalue()
    else:
        yaml.dump(data=data, stream=stream)
