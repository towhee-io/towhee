# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from towhee.utils.sqlalchemy_utils import Session
from towhee.utils.thirdparty.fastapi_utils import FastAPI, Depends, HTTPException
from . import crud, models, schemas
from .database import SQLDataBase

db_obj = SQLDataBase()
engine = db_obj.engine
SessionLocal = db_obj.SessionLocal

models.Base.metadata.create_all(bind=engine)

app = FastAPI()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post('/pipeline/create')
def create_pipeline(pipeline: schemas.PipelineCreate, db: Session = Depends(get_db)):
    try:
        p = crud.get_pipeline_by_name(db, name=pipeline.name)
        if p:
            raise HTTPException(status_code=400, detail=f'Pipeline: {pipeline.name} already exists, please rename it.')
        meta = crud.create_pipeline_meta(db, meta=pipeline)

        pipeline_info = schemas.PipelineUpdate(name=pipeline.name, dag_json_str=pipeline.dag_json_str)
        _ = crud.create_pipeline_info(db, info=pipeline_info, meta_id=meta.id, version=0)
        return {'status': 200, 'msg': 'Successfully create pipeline.'}
    except HTTPException as e:
        return {'status': e.status_code, 'msg': e.detail}
    except Exception as e:  # pylint: disable=broad-except
        return {'status': 400, 'msg': str(e)}


@app.post('/pipeline/update')
def update_pipeline(pipeline: schemas.PipelineUpdate, db: Session = Depends(get_db)):
    try:
        meta = crud.get_pipeline_by_name(db, name=pipeline.name)
        if not meta:
            raise HTTPException(status_code=404, detail=f'Pipeline: {pipeline.name} not exist, please create it first.')
        count = crud.count_pipeline_by_name(db, name=pipeline.name)
        _ = crud.create_pipeline_info(db, info=pipeline, meta_id=meta.id, version=count)
        return {'status': 200, 'msg': 'Successfully update pipeline.'}
    except HTTPException as e:
        return {'status': e.status_code, 'msg': e.detail}
    except Exception as e:  # pylint: disable=broad-except
        return {'status': 400, 'msg': str(e)}


@app.get('/pipelines')
def get_pipelines(db: Session = Depends(get_db)):
    try:
        pipe_list = crud.get_pipeline_list(db)
        if not pipe_list:
            raise HTTPException(status_code=404, detail='There is no pipeline.')
        result = dict((p[0], p[1]) for p in pipe_list)
        return {'status': 200, 'msg': result}
    except HTTPException as e:
        return {'status': e.status_code, 'msg': e.detail}
    except Exception as e:  # pylint: disable=broad-except
        return {'status': 400, 'msg': str(e)}


@app.get('/{pipeline_name}/info')
def get_pipeline_info(pipeline_name: str, db: Session = Depends(get_db)):
    try:
        infos = crud.get_pipeline_info_by_name(db, name=pipeline_name)
        if not infos:
            raise HTTPException(status_code=404, detail=f'Pipeline: {pipeline_name} not exist, please create it first.')
        result = dict((i[0], i[1]) for i in infos)
        return {'status': 200, 'msg': result}
    except HTTPException as e:
        return {'status': e.status_code, 'msg': e.detail}
    except Exception as e:  # pylint: disable=broad-except
        return {'status': 400, 'msg': str(e)}


@app.get('/{pipeline_name}/{version}/dag')
def get_pipeline_dag(pipeline_name: str, version: int, db: Session = Depends(get_db)):
    try:
        dag = crud.get_pipeline_dag_by_name_version(db, name=pipeline_name, version=version)
        if not dag:
            raise HTTPException(status_code=404, detail=f'Pipeline: {pipeline_name} not exist, please create it first.')
        return {'status': 200, 'msg': dag}
    except HTTPException as e:
        return {'status': e.status_code, 'msg': e.detail}
    except Exception as e:  # pylint: disable=broad-except
        return {'status': 400, 'msg': str(e)}


@app.delete('/{pipeline_name}')
def delete_pipeline(pipeline_name: str, db: Session = Depends(get_db)):
    try:
        meta = crud.get_pipeline_by_name(db, name=pipeline_name)
        if not meta:
            raise HTTPException(status_code=404, detail=f'Pipeline: {pipeline_name} not exist.')
        crud.delete_pipeline(db, meta_id=meta.id)
        return {'status': 200, 'msg': 'Successfully deleted the pipeline.'}
    except HTTPException as e:
        return {'status': e.status_code, 'msg': e.detail}
    except Exception as e:  # pylint: disable=broad-except
        return {'status': 400, 'msg': str(e)}
