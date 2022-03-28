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

from towhee.utils.repo_normalize import RepoNormalize


class URI:
    """_summary_

    Examples:

    >>> op = URI('resnet-image-embedding')
    >>> op.namespace
    'towhee'
    >>> op.repo
    'resnet-image-embedding'
    >>> op.module_name
    'resnet_image_embedding'
    >>> op.resolve_module('my')
    'my/resnet_image_embedding'
    >>> op.resolve_modules('my1', 'my2')
    ['my1/resnet_image_embedding', 'my2/resnet_image_embedding']
    """

    def __init__(self, uri: str) -> None:
        self._raw = uri
        result = RepoNormalize(uri).parse_uri()
        for field in result._fields:
            setattr(self, field, getattr(result, field))

    @property
    def namespace(self):
        return self.author.replace('_', '-')

    @property
    def short_uri(self):
        return self.namespace + '/' + self.norm_repo

    @property
    def full_name(self):
        return self.namespace + '/' + self.module_name

    def resolve_module(self, ns):
        if not self.has_ns:
            self.author = ns
        return self.full_name

    def resolve_modules(self, *arg):
        return [self.resolve_module(ns) for ns in arg]

    def resolve_repo(self, ns):
        if not self.has_ns:
            self.author = ns
        return self.short_uri

    def resolve_repos(self, *arg):
        return [self.resolve_repo(ns) for ns in arg]


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)
