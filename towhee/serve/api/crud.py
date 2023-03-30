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

from towhee.utils.sqlalchemy_utils import Session, func

from . import models, schemas


def get_pipeline_list(db: Session, skip: int = 0, limit: int = 1000):
    return db.query(models.Meta.name, models.Meta.description).filter(models.Meta.state == 0).offset(skip).limit(limit).all()


def get_pipeline_by_name(db: Session, name: str):
    return db.query(models.Meta).filter(models.Meta.name == name).filter(models.Meta.state == 0).first()


def get_pipeline_info_by_name(db: Session, name: str):
    res = db.query(models.Info.version, models.Info.date).join(models.Meta).filter(models.Meta.name == name).filter(models.Meta.state == 0) \
            .filter(models.Info.state == 0).all()
    return res


def get_pipeline_dag_by_name_version(db: Session, name: str, version: int):
    res = db.query(models.Info.dag_json_str).join(models.Meta).filter(models.Meta.name == name).filter(models.Meta.state == 0) \
            .filter(models.Info.state == 0).filter(models.Info.version == version).scalar()
    return res


def get_pipeline_dag_by_name(db: Session, name: str, version: str):
    res = db.query(models.Meta).filter(models.Meta.name == name).filter(models.Meta.state == 0).first() \
            .filter(models.Info.state == 0).filter(models.Info.version == version).first()
    return res.info.dag_json_str


def count_pipeline_by_name(db: Session, name: str):
    res = db.query(func.count(models.Info.id)).join(models.Meta).filter(models.Meta.name == name).filter(models.Meta.state == 0) \
            .filter(models.Info.state == 0).scalar()
    return res


def delete_pipeline(db: Session, meta_id: str):
    db.query(models.Meta).filter(models.Meta.id == meta_id).update({'state': 1})
    db.commit()

    db.query(models.Info).filter(models.Info.meta_id == meta_id).update({'state': 1})
    db.commit()


def create_pipeline_meta(db: Session, meta: schemas.PipelineCreate, state: int = 0):
    db_meta = models.Meta(name=meta.name, description=meta.description, state=state)
    db.add(db_meta)
    db.commit()
    db.refresh(db_meta)
    return db_meta


def create_pipeline_info(db: Session, info: schemas.PipelineUpdate, meta_id: int, version: int, state: int = 0):
    db_info = models.Info(meta_id=meta_id, version=version, dag_json_str=info.dag_json_str, state=state)
    db.add(db_info)
    db.commit()
    db.refresh(db_info)
    return db_info
