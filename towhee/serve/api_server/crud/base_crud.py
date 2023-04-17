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

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession


class BaseCRUD:
    """
    Base CRUD
    """
    def __init__(self, model):
        self.model = model

    async def create(self, db: AsyncSession, **kwargs):
        model_obj = self.model(**kwargs)
        db.add(model_obj)
        await db.commit()
        await db.refresh(model_obj)
        return model_obj

    async def update_state(self, db: AsyncSession, model_id: int, state: int = 1):
        stmt = update(self.model).filter(self.model.id == model_id).values(state=state)
        await db.execute(stmt)
        await db.commit()
