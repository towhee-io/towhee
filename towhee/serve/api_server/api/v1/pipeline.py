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

from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, APIRouter

from towhee.serve.api_server.crud import pipeline_meta, pipeline_info
from towhee.serve.api_server.api.deps import get_db
from towhee.serve.api_server.schemas.pipeline_schemas import PipelineCreate, PipelineUpdate
from towhee.serve.api_server.schemas.return_schemas import ReturnBase

router = APIRouter()


# pylint: disable=broad-except
@router.post('/create', response_model=ReturnBase)
async def create_pipeline(pipeline: PipelineCreate, db: Session = Depends(get_db)):
    try:
        p = await pipeline_meta.get_by_name(db, name=pipeline.name)
        if p:
            raise HTTPException(status_code=400, detail=f'Pipeline: {pipeline.name} already exists, please rename it.')
        meta = await pipeline_meta.create(db, name=pipeline.name, description=pipeline.description)

        await pipeline_info.create(db, meta_id=meta.id, dag_json_str=pipeline.dag_json_str, version=0)
        return ReturnBase(status_code=0, msg='Successfully create pipeline.')
    except HTTPException as e:
        return ReturnBase(status_code=-1, msg=e.detail)
    except Exception as e:
        return ReturnBase(status_code=-1, msg=str(e))


@router.post('/update', response_model=ReturnBase)
async def update_pipeline(pipeline: PipelineUpdate, db: Session = Depends(get_db)):
    try:
        meta = await pipeline_meta.get_by_name(db, name=pipeline.name)
        if not meta:
            raise HTTPException(status_code=404, detail=f'Pipeline: {pipeline.name} not exist, please create it first.')
        count = await pipeline_info.count_pipeline_by_name(db, name=pipeline.name)
        await pipeline_info.create(db, meta_id=meta[0].id, dag_json_str=pipeline.dag_json_str, version=count)
        return ReturnBase(status_code=0, msg='Successfully update pipeline.')
    except HTTPException as e:
        return ReturnBase(status_code=-1, msg=e.detail)
    except Exception as e:
        return ReturnBase(status_code=-1, msg=str(e))


@router.get('/list', response_model=ReturnBase)
async def get_pipelines(db: Session = Depends(get_db)):
    try:
        pipe_list = await pipeline_meta.get_list(db)
        if not pipe_list:
            raise HTTPException(status_code=404, detail='There is no pipeline.')
        result = dict((p[0], p[1]) for p in pipe_list)
        return ReturnBase(status_code=0, msg='Successfully list all pipelines.', data=result)
    except HTTPException as e:
        return ReturnBase(status_code=-1, msg=e.detail)
    except Exception as e:
        return ReturnBase(status_code=-1, msg=str(e))


@router.get('/{pipeline_name}/info', response_model=ReturnBase)
async def get_pipeline_info(pipeline_name: str, db: Session = Depends(get_db)):
    try:
        infos = await pipeline_info.get_by_name(db, name=pipeline_name)
        if not infos:
            raise HTTPException(status_code=404, detail=f'Pipeline: {pipeline_name} not exist, please create it first.')
        result = dict((i[0].version, i[0].date) for i in infos)
        return ReturnBase(status_code=0, msg=f'Successfully get the {pipeline_name} info.', data=result)
    except HTTPException as e:
        return ReturnBase(status_code=-1, msg=e.detail)
    except Exception as e:
        return ReturnBase(status_code=-1, msg=str(e))


@router.get('/{pipeline_name}/{version}/dag', response_model=ReturnBase)
async def get_pipeline_dag(pipeline_name: str, version: int, db: Session = Depends(get_db)):
    try:
        dag = await pipeline_info.get_dag_by_name_version(db, name=pipeline_name, version=version)
        if not dag:
            raise HTTPException(status_code=404, detail=f'Pipeline: {pipeline_name} not exist, please create it first.')
        return ReturnBase(status_code=0, msg=f'Successfully get the dag for {pipeline_name} with version {version}.', data={'dag_str': dag})
    except HTTPException as e:
        return ReturnBase(status_code=-1, msg=e.detail)
    except Exception as e:
        return ReturnBase(status_code=-1, msg=str(e))


@router.delete('/{pipeline_name}', response_model=ReturnBase)
async def delete_pipeline(pipeline_name: str, db: Session = Depends(get_db)):
    try:
        meta = await pipeline_meta.get_by_name(db, name=pipeline_name)
        if not meta:
            raise HTTPException(status_code=404, detail=f'Pipeline: {pipeline_name} not exist.')
        await pipeline_meta.update_state(db, model_id=meta[0].id)
        await pipeline_info.update_state(db, model_id=meta[0].id)
        return ReturnBase(status_code=0, msg=f'Successfully delete the {pipeline_name}.')
    except HTTPException as e:
        return ReturnBase(status_code=-1, msg=e.detail)
    except Exception as e:
        return ReturnBase(status_code=-1, msg=str(e))
