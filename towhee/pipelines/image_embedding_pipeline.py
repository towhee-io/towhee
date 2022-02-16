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
from typing import List, Union

import yaml

from towhee.pipelines.base import PipelineBase
from towhee import Build, pipeline


class ImageEmbeddingPipeline(PipelineBase):
    """
    Pipeline for image embedding tasks.

    Args:
        model: (`str` or `List[str]`)
            Specifies the model used for image embedding. The user can pass a list
            of model names to create a pipeline ensembling multiple models.

            Supported models:
                `vgg`,
                `resnet50`, `resnet101`,
                `swin-transformer`,
                `vit`,
                ...

        ensemble: (`str`)
            Specifies the type of model ensemble. This argument works iff
            multiple model names are given via `model`.

            Supported ensemble types:
                `linear`,
    """

    def __init__(self,
                 model: Union[str, List[str]] = None,
                 ensemble: str = None):
        models = []
        if isinstance(model, str):
            models = [model]
        else:
            models = model
        num_branch = len(models)

        models = dict(
            zip([
                'embedding_model_1',
                'embedding_model_2',
                'embedding_model_3',
            ], models))
        if ensemble:
            models['ensemble_model'] = ensemble
        self._pipeline = Build(**models).pipeline(
            'builtin/image_embedding_template_{}'.format(num_branch))

    def __call__(self, *arg, **kws):
        return self._pipeline(*arg, **kws)

    def save(self, name: str, path: str = '.'):
        operator_path = path + '/' + name
        if os.path.exists(operator_path):
            raise FileExistsError(operator_path)
        os.mkdir(operator_path)
        with open('{}/{}.yaml'.format(operator_path, name), 'w', encoding='utf-8') as f:
            info = yaml.safe_load(self._pipeline.pipeline.graph_repr.ir)
            info['name'] = name
            f.write(yaml.safe_dump(info))

    def push_to_hub(self, version: str = 'main'):
        # TODO: push to hub with new hub tool
        pass


def image_embedding_pipeline(model: Union[str, List[str]] = None,
                             ensemble: str = None,
                             name: str = None,
                             version: str = None):
    pipe = None
    if name is not None:
        pipe = pipeline(name, tag=version)
        return pipe
    if model is not None:
        return ImageEmbeddingPipeline(model=model, ensemble=ensemble)
