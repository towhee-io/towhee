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

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .base_crud import BaseCRUD

from towhee.serve.api_server.models.pipeline_meta import PipelineMeta
from towhee.serve.api_server.models.pipeline_info import PipelineInfo


class PipelineInfoCRUD(BaseCRUD):
    """
    pipeline_info CRUD
    """
    async def get_by_name(self, db: AsyncSession, name: str):
        stmt = select(self.model).join(PipelineMeta).filter(PipelineMeta.name == name).filter(PipelineMeta.state == 0) \
                .filter(self.model.state == 0)
        res = await db.execute(stmt)
        return res.all()

    async def get_dag_by_name_version(self, db: AsyncSession, name: str, version: int):
        stmt = select(self.model.dag_json_str).join(PipelineMeta).filter(PipelineMeta.name == name).filter(PipelineMeta.state == 0) \
                .filter(self.model.state == 0).filter(self.model.version == version)
        res = await db.execute(stmt)
        return res.scalar()

    # pylint: disable=not-callable
    async def count_pipeline_by_name(self, db: AsyncSession, name: str):
        stmt = select(func.count(self.model.id)).join(PipelineMeta).filter(PipelineMeta.name == name).filter(PipelineMeta.state == 0) \
                .filter(self.model.state == 0)
        res = await db.execute(stmt)
        return res.scalar()

    async def update_state(self, db: AsyncSession, model_id: int, state: int = 1):
        stmt = update(self.model).filter(self.model.meta_id == model_id).values(state=state)
        await db.execute(stmt)
        await db.commit()


pipeline_info = PipelineInfoCRUD(PipelineInfo)
