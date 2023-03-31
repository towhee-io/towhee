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

from sqlalchemy.orm import Session

from .base_crud import BaseCRUD
from towhee.serve.api_server.models.pipeline_meta import PipelineMeta


class PipelineMetaCRUD(BaseCRUD):
    def get_list(self, db: Session, skip: int = 0, limit: int = 1000):
        return db.query(self.model.name, self.model.description).filter(self.model.state == 0).offset(skip).limit(limit).all()

    def get_by_name(self, db: Session, name: str):
        return db.query(self.model).filter(self.model.name == name).filter(self.model.state == 0).first()


pipeline_meta = PipelineMetaCRUD(PipelineMeta)
