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

import os
import sys
import yaml
from typing import Union
from pathlib import Path
from shutil import copyfile, copytree, rmtree
from importlib import import_module

import git
from towhee.utils.log import engine_log


def convert_dict(dicts: dict) -> dict:
    """
    Convert all the values in a dictionary to str and replace char.

    For example:
    <class 'torch.Tensor'>(unknow type) to torch.Tensor(str type).

    Args:
        dicts (`dict`):
            The dictionary to convert.

    Returns:
        (`dict`)
            The converted dictionary.
    """
    for keys in dicts:
        dicts[keys] = str(dicts[keys]).replace('<class ', '').replace('>', '').replace('\'', '')
    return dict(dicts)


def copy_files(src: Union[str, Path], dst: Union[str, Path], replace: bool):
    """
    Copy files in src dir to dst dir.

    Args:
        src (`Union[str, Path]`):
            The source dir.
        dst (`Union[str, Path]`):
            The destination dir.
        replace (`bool`):
            Whether to replace the file with same name.
    """
    src = Path(src)
    dst = Path(dst)

    for f in os.listdir(src):
        if (dst / f).exists():
            if replace:
                rmtree(dst / f)
            else:
                continue
        if (src / f).is_file():
            copyfile(src / f, dst / f)
        elif (src / f).is_dir():
            copytree(src / f, dst / f)


def init_operator(
    author: str, repo: str, is_nn: bool, file_src: Union[str, Path], file_dst: Union[str, Path] = None, root: str = 'https://hub.towhee.io'
) -> None:
    """
    Initialize the repo with template.

    First clone the repo, then move and rename the template repo file.

    Args:
        author (`str`):
            The author of the repo.
        repo (`str`):
            The name of the repo.
        is_nn (`bool`):
            If the operator is an nnoperator(neural network related).
        file_src (`Union[str, Path]`):
            The path to the template files.
        file_dst (`Union[str, Path]`):
            The path to the local repo to init.
        root (`str`):
            The root url where the repo located.

    Raises:
        (`HTTPError`)
            Raise error in request.
        (`OSError`)
            Raise error in writing file.
    """
    repo_file_name = repo.replace('-', '_')

    if not file_dst:
        file_dst = Path().cwd() / repo_file_name
    file_src = Path(file_src)
    file_dst = Path(file_dst)

    url = root + '/' + author + '/' + repo + '.git'
    git.Repo.clone_from(url=url, to_path=file_dst, branch='main')

    if is_nn:
        template = 'nnoperator_template'
    else:
        template = 'pyoperator_template'

    copy_files(src=file_src, dst=file_dst, replace=False)

    (file_dst / (template + '.py')).rename(file_dst / (repo_file_name + '.py'))
    (file_dst / (template + '.yaml')).rename(file_dst / (repo_file_name + '.yaml'))


def init_pipeline(author: str, repo: str, file_src: Union[str, Path], file_dst: Union[str, Path] = None, root: str = 'https://hub.towhee.io') -> None:
    """
    Initialize the repo with template.

    First clone the repo, then move and rename the template repo file.

    Args:
        author (`str`):
            The author of the repo.
        repo (`str`):
            The name of the repo.
        file_src (`Union[str, Path]`):
            The path to the template files.
        file_dst (`Union[str, Path]`):
            The path to the local repo to init.
        root (`str`):
            The root url where the repo located.

    Raises:
        (`HTTPError`)
            Raise error in request.
        (`OSError`)
            Raise error in writing file.
    """
    repo_file_name = repo.replace('-', '_')

    if not file_dst:
        file_dst = Path.cwd() / repo_file_name
    file_src = Path(file_src)
    file_dst = Path(file_dst)

    url = root + '/' + author + '/' + repo + '.git'
    git.Repo.clone_from(url=url, to_path=file_dst, branch='main')

    copy_files(src=file_src, dst=file_dst, replace=False)

    (file_dst / 'pipeline_template.yaml').rename(file_dst / (repo_file_name + '.yaml'))


def generate_yaml(author: str, repo: str, local_dir: Union[str, Path] = None) -> None:
    """
    Generate the yaml of Operator.

    Raises:
        (`HTTPError`)
            Raise error in request.
        (`OSError`)
            Raise error in writing file.
    """
    if not local_dir:
        local_dir = Path.cwd() / repo.replace('-', '_')
    local_dir = Path(local_dir).resolve()
    sys.path.append(str(local_dir))

    yaml_file = local_dir / (repo.replace('-', '_') + '.yaml')
    if yaml_file.exists():
        engine_log.error('%s already exists in %s.', yaml_file, local_dir)
        return

    class_name = ''.join(x.title() for x in repo.split('-'))
    author_operator = author + '/' + repo

    # import the class from repo
    cls = getattr(import_module(repo.replace('-', '_')), class_name)
    init_args = cls.__init__.__annotations__
    try:
        del init_args['return']
    except KeyError:
        pass
    call_func = cls.__call__.__annotations__
    try:
        call_output = call_func.pop('return')
        call_output = call_output.__annotations__
    except KeyError:
        pass

    data = {
        'name': repo,
        'labels': {
            'recommended_framework': '', 'class': '', 'others': ''
        },
        'operator': author_operator,
        'init': convert_dict(init_args),
        'call': {
            'input': convert_dict(call_func), 'output': convert_dict(call_output)
        }
    }
    with open(yaml_file, 'a', encoding='utf-8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)
