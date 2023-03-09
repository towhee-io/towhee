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


from towhee import ops, pipe, AutoPipes, AutoConfig


@AutoConfig.register
class MilvusInsertConfig:
    """
    Config of pipeline
    """
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = '19530'
        self.collection_name = None
        self.user = None
        self.password = None


@AutoPipes.register
def milvus_insert_pipe(config):
    return (
        pipe.input('row')
        .map('row', 'mr', ops.ann_insert.milvus_client(host=config.host,
                                                     port=config.port,
                                                     collection_name=config.collection_name,
                                                     user=config.user,
                                                     password=config.password
                                                     ))
        .output('mr')
    )
