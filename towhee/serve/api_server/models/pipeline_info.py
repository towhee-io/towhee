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
from datetime import datetime
from sqlalchemy import Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship

from .base import Base


class PipelineInfo(Base):
    """
    pipeline_info Table
    """
    __tablename__ = 'pipeline_info'

    meta_id = Column(Integer, ForeignKey('pipeline_meta.id'))
    version = Column(Integer, nullable=False)
    dag_json_str = Column(String, nullable=False)
    date = Column(DateTime, default=datetime.now)

    meta = relationship('PipelineMeta', back_populates='info')
