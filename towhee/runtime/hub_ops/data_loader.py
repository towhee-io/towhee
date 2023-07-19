# Copyright 2023 Zilliz. All rights reserved.
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

from typing import Any
from towhee.runtime.factory import HubOp


class DataLoader:
    """
    The data loader Operator is used to load data from various sources.
    """

    doc_loader: HubOp = HubOp('data_loader.doc_loader')
    """
    Load text from doc/docx files.

    __init__(self)

    __call__(self, path: str):
        path(`str`):
            The path to the doc/docx file.

    Example:

    .. code-block:: python

        from glob import glob
        from towhee import ops, pipe

        p = (
            pipe.input('path')
                .map('path', 'text', ops.data_loader.doc_loader())
                .output('text')
        )

        files = glob('./doc/*.doc*')

        for file in files:
            p(file)
    """

    excel_loader: HubOp = HubOp('data_loader.excel_loader')
    """
    Load text from excel files.

    __init__(self)

    __call__(self, path: str):
        path(`str`):
            The path to the excel file.

    Example:

    .. code-block:: python

        from glob import glob
        from towhee import ops, pipe

        p = (
            pipe.input('path')
                .map('path', 'text', ops.data_loader.excel_loader())
                .output('text')
        )

        files = glob('./excel/*.xls*')

        for file in files:
            p(file)
    """

    markdown_loader: HubOp = HubOp('data_loader.markdown_loader')
    """
    Load text from markdown files.

    __init__(self)

    __call__(self, path: str):
        path(`str`):
            The path to the markdown file.

    Example:

    .. code-block:: python

        from glob import glob
        from towhee import ops, pipe

        p = (
            pipe.input('path')
                .map('path', 'text', ops.data_loader.markdown_loader())
                .output('text')
        )

        files = glob('./markdown/*.md')

        for file in files:
            p(file)
    """

    pdf_loader: HubOp = HubOp('data_loader.pdf_loader')
    """
    Load text from pdf files.

    __init__(self)

    __call__(self, path: str, password: Optional[Union[str, bytes]] = None):
        path(`str`):
            The path to the pdf file.
        password(`Optional[Union[str, bytes]]`):
            The password for the pdf file if required.

    Example:

    .. code-block:: python

        from glob import glob
        from towhee import ops, pipe

        p = (
            pipe.input('path')
                .map('path', 'text', ops.data_loader.pdf_loader())
                .output('text')
        )

        files = glob('./pdf/*.pdf')

        for file in files:
            p(file)
    """

    text_loader: HubOp = HubOp('data_loader.text_loader')
    """
    Load text from pure text files, e.g. `.txt`, `.py`, `.csv`, etc..

    __init__(self, encoding: str = None):
        encoding(`str`):
            The file encoding to use.

    __call__(self, path: str):
        path(`str`):
            The path to the text file.

    Example:

    .. code-block:: python

        from glob import glob
        from towhee import ops, pipe

        p = (
            pipe.input('path')
                .map('path', 'text', ops.data_loader.text_loader(encoding='utf-8'))
                .output('text')
        )

        files = glob('./text/*.txt')
        file.extend(glob('./python/*.py'))

        for file in files:
            p(file)
    """

    html_loader: HubOp = HubOp('data_loader.html_loader')
    """
    Load text from html, either from local files or url.

    __init__(self, encoding: str = None):
        encoding(`str`):
            The html encoding to use.

    __call__(self, path: str, encoding: str = None):
        path(`str`):
            The path to the text file.
        encoding(`str`):
            The html encoding to use.

    Example:

    .. code-block:: python

        from glob import glob
        from towhee import ops, pipe

        p = (
            pipe.input('path')
                .map('path', 'text', ops.data_loader.html_loader(encoding='utf-8'))
                .output('text')
        )

        files = glob('./html/*.html')

        for file in files:
            p(file)
    """

    notebook_loader: HubOp = HubOp('data_loader.notebook_loader')
    """
    Load text from jupyter notebook files.

    __init__(self, max_output_length: int = None):
        max_output_length(`str`):
            The max length of cell outputs.

    __call__(self, path: str):
        path(`str`):
            The path to the notebook file.

    Example:

    .. code-block:: python

        from glob import glob
        from towhee import ops, pipe

        p = (
            pipe.input('path')
                .map('path', 'text', ops.data_loader.notebook_loader(max_output_length=100))
                .output('text')
        )

        files = glob('./notebook/*.ipynb')

        for file in files:
            p(file)
    """

    powerpoint_loader: HubOp = HubOp('data_loader.powerpoint_loader')
    """
    Load text from powerpoint files.

    __init__(self)

    __call__(self, path: str):
        path(`str`):
            The path to the powerpoint file.

    Example:

    .. code-block:: python

        from glob import glob
        from towhee import ops, pipe

        p = (
            pipe.input('path')
                .map('path', 'text', ops.data_loader.powerpoint_loader())
                .output('text')
        )

        files = glob('./powerpoint/*.pptx')

        for file in files:
            p(file)
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return HubOp('towhee.data_loader')(*args, **kwds)
