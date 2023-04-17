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

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from towhee.serve.api_server.config import SQL_URL

engine = create_async_engine(SQL_URL,
                             echo=True,
                             )

# pylint: disable=invalid-name
SessionLocal = sessionmaker(autocommit=False,
                            autoflush=False,
                            bind=engine,
                            class_=AsyncSession,
                            expire_on_commit=False)
