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


from towhee.runtime.factory import HubOp


class DataSource:
    """
    `data_source <https://towhee.io/data-source?index=1&size=30&type=2>`_ load data from many different sources.
    Work with `DataLoader <https://towhee.readthedocs.io/en/latest/data_source/data_source.html>`_
    """

    glob: HubOp = HubOp('data_source.glob')
    """
    `glob <https://towhee.io/data-source/glob>`_ wrapper of python glob.glob
    Return a list of paths matching a pathname pattern.
    The pattern may contain simple shell-style wildcards a la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.

    If recursive is true, the pattern '**' will match any files and
    zero or more directories and subdirectories.

    __init__(self, pathname, *, recursive=False):
        pathname(`str`):
            path pattern.
        recursive(`bool`):
            Default is False

    Example:

    .. code-block:: python

        from towhee import DataLoader, pipe, ops
        p = (
            pipe.input('image_path')
            .map('image_path', 'image', ops.image_decode.cv2())
            .map('image', 'vec', ops.image_embedding.timm(model_name='resnet50'))
            .output('vec')

        )

        for data in DataLoader(ops.data_source.glob('./*.jpg')):
            print(p(data).to_list(kv_format=True))

        # batch
        for data in DataLoader(ops.data_source.glob('./*.jpg'), batch_size=10):
            p.batch(data)
    """

    csv_reader: HubOp = HubOp('data_source.csv_reader')
    """
    `csv_reader <https://towhee.io/data-source/csv-reader>`_ Wrapper of python csv:
    https://docs.python.org/3.8/library/csv.html .

    __init__(self, f_path: str, dialect='excel', **fmtparams):
        csvfile(`str`):
            csvfile path

    Example:

    .. code-block:: python

        from towhee import DataLoader, pipe, ops
        p = (
            pipe.input('image_path')
            .map('image_path', 'image', ops.image_decode.cv2())
            .map('image', 'vec', ops.image_embedding.timm(model_name='resnet50'))
            .output('vec')

        )

        # csv data format: id,image_path,label
        for data in DataLoader(ops.data_source.csv_reader('./reverse_image_search.csv'), parser=lambda x: x[1]):
            print(p(data).to_list(kv_format=True))

        # batch
        for data in DataLoader(ops.data_source.csv_reader('./reverse_image_search.csv'), parser=lambda x: x[1], batch_size=10):
            p.batch(data)
    """

    sql: HubOp = HubOp('data_source.sql')
    """
    `sql <https://towhee.io/data-source/sql>` read data from sqlite or mysql.

    __init__(self, sql_url: str, table_name:str, cols: str = '*', where: str = None, limit: int = 500):
        sql_url(`str`):
            the url of the sql database for cache, such as '<db_type>+<db_driver>://:@:/'
                sqlite: sqlite:///./sqlite.db
                mysql: mysql+pymysql://root:123456@127.0.0.1:3306/mysql

        table_name(`str`):
            table

        cols(`str`):
            The columns to be queried, default to *, indicating all columns
            If you want to query specific columns, use the column names and separate them with ',',
            such as 'id,image_path,label'.

        where('str`):
            Where conditional statement, for example: id > 100

        limit(`int`):
            The default value is 500. If set to None, all data will be returned.

    Example:

    .. code-block:: python

        from towhee import DataLoader, pipe, ops
        p = (
            pipe.input('image_path')
            .map('image_path', 'image', ops.image_decode.cv2())
            .map('image', 'vec', ops.image_embedding.timm(model_name='resnet50'))
            .output('vec')

        )

        # table cols: id, image_path, label

        for data in DataLoader(ops.data_source.sql('sqlite:///./sqlite.db', 'image_table'), parser=lambda x: x[1]):
            print(p(data).to_list(kv_format=True))

        # batch
        for data in DataLoader(ops.data_source.sql('sqlite:///./sqlite.db', 'image_table'), parser=lambda x: x[1], batch_size=10):
            p.batch(data)
    """

    readthedocs: HubOp = HubOp('data_source.readthedocs')
    """
    `readthedocs <https://towhee.io/data-source/readthedocs>`_ to get the list of documents for a single Read the Docs project.

    __init__(self, page_prefix: str, index_page: str = None, include: Union[List[str], str] = '', exclude: Union[List[str], str] = None):
        page_prefix(`str`):
            The root path of the page. Generally, the crawled links are relative paths.
            The complete URL needs to be obtained by splicing the root path + relative path.

        index_page(`str`):
            The main page contains links to all other pages, if None, will use page_prefix.
            example: https://towhee.readthedocs.io/en/latest/

        include(`Union[List[str], str]`):
            Only contains URLs that meet this condition.

        exclude(`Union[List[str], str]`):
            Filter out URLs that meet this condition.

    Example:

    .. code-block:: python

        from towhee import DataLoader, pipe, ops
        p = (
            pipe.input('url')
            .map('url', 'text', ops.text_loader())
            .flat_map('text', 'sentence', ops.text_splitter())
            .map('sentence', 'embedding', ops.sentence_embedding.transformers(model_name='all-MiniLM-L6-v2'))
            .map('embedding', 'embedding', ops.towhee.np_normalize())
            .output('embedding')
        )



        for data in DataLoader(ops.data_source.readthedocs('https://towhee.readthedocs.io/en/latest/',
                                                           include='html', exclude='index.html')):
            print(p(data).to_list(kv_format=True))

        # batch
        for data in DataLoader(ops.data_source.readthedocs('https://towhee.readthedocs.io/en/latest/',
                                                           include='html', exclude='index.html'), batch_size=10):
            p.batch(data)
    """

    def __call__(self, *args, **kwargs):
        return HubOp('towhee.data_source')(*args, **kwargs)

