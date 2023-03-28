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


class BaseCRUD:
    """
    Base CRUD
    """
    def __init__(self, model):
        self.model = model

    def create(self, db, **kwargs):
        model_obj = self.model(**kwargs)
        db.add(model_obj)
        db.commit()
        db.refresh(model_obj)
        return model_obj

    def update_state(self, db, model_id: int, state: int = 1):
        db.query(self.model).filter(self.model.id == model_id).update({'state': state})
        db.commit()
