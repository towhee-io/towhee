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

from sqlalchemy import func
from sqlalchemy.orm import Session

from .base_crud import BaseCRUD

from towhee.serve.api_server.models.pipeline_meta import PipelineMeta
from towhee.serve.api_server.models.pipeline_info import PipelineInfo


class PipelineInfoCRUD(BaseCRUD):
    """
    pipeline_info CRUD
    """
    def get_by_name(self, db: Session, name: str):
        res = db.query(self.model).join(PipelineMeta).filter(PipelineMeta.name == name).filter(PipelineMeta.state == 0) \
                .filter(self.model.state == 0).all()
        return res

    def get_dag_by_name_version(self, db: Session, name: str, version: int):
        res = db.query(self.model.dag_json_str).join(PipelineMeta).filter(PipelineMeta.name == name).filter(PipelineMeta.state == 0) \
                .filter(self.model.state == 0).filter(self.model.version == version).scalar()
        return res

    def count_pipeline_by_name(self, db: Session, name: str):
        res = db.query(func.count(self.model.id)).join(PipelineMeta).filter(PipelineMeta.name == name).filter(PipelineMeta.state == 0) \
                .filter(self.model.state == 0).scalar()
        return res

    def update_state(self, db: Session, model_id: int, state: int = 1):
        db.query(self.model).filter(self.model.meta_id == model_id).update({'state': state})
        db.commit()


pipeline_info = PipelineInfoCRUD(PipelineInfo)
