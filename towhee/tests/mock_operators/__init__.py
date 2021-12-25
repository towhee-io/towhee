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

import os
import sys
import importlib
import logging

_MOCK_OPERATOR_DIR = os.path.dirname(__file__)

ADD_OPERATOR_PATH = os.path.join(_MOCK_OPERATOR_DIR, 'add_operator')
SUB_OPERATOR_PATH = os.path.join(_MOCK_OPERATOR_DIR, 'sub_operator')
PYTORCH_CNN_OPERATOR_PATH = os.path.join(_MOCK_OPERATOR_DIR, 'pytorch_cnn_operator')
PYTORCH_IMAGE_CLASSIFICATION_OPERATOR_PATH = os.path.join(_MOCK_OPERATOR_DIR, 'pytorch_image_classification_operator')
PYTORCH_OBJECT_DETECTION_OPERATOR_PATH = os.path.join(_MOCK_OPERATOR_DIR, 'pytorch_object_detection_operator')
PYTORCH_TRANSFORMER_OPERATOR_PATH = os.path.join(_MOCK_OPERATOR_DIR, 'pytorch_transformer_operator')
PYTORCH_TRANSFORM_OPERATOR_PATH = os.path.join(_MOCK_OPERATOR_DIR, 'pytorch_transform_operator')
PYTORCH_TRANSFORMER_VEC_OPERATOR_PATH = os.path.join(_MOCK_OPERATOR_DIR, 'pytorch_transformer_vec_operator')


def load_local_operator(op_name: str, op_path: str):
    """
    A simple operator_loader function for UT
    """
    try:
        sys.path.insert(0, op_path)
        return importlib.import_module(op_name)
    except ModuleNotFoundError as e:
        logging.error('import model: {op_name} from path: {op_path} failed, error: {err}', op_name=op_name, op_path=op_path, err=str(e))
    finally:
        sys.path.pop(0)
