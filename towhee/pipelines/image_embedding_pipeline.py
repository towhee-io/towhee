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

from typing import Any, List, Union

import yaml
from pathlib import Path
from towhee.hparam.hyperparameter import param_scope
from towhee.pipelines.alias_resolvers import get_resolver
from towhee.pipelines.base import PipelineBase
from towhee import Inject, pipeline


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
    def __init__(self, model: Union[Any, List[Any]] = None, ensemble: str = None):
        with param_scope() as hp:
            resolver = get_resolver(hp().towhee.alias_resolver('local'))

        models: List[Any] = []
        if isinstance(model, str):
            models = [model]
        else:
            models = model
        num_branch = len(models)

        models = [resolver.resolve(model) if isinstance(model, str) else model for model in models]
        operators = dict(zip([
            'embedding_model_1',
            'embedding_model_2',
            'embedding_model_3',
        ], models))
        if ensemble is not None:
            operators['ensemble_model'] = resolver.resolve(ensemble) if isinstance(ensemble, str) else ensemble

        injections = {name: {'function': model.function, 'init_args': model.init_args} for name, model in operators.items()}
        self._pipeline = Inject(**injections).pipeline('builtin/image_embedding_template_{}'.format(num_branch))

    def __call__(self, *arg, **kws):
        return self._pipeline(*arg, **kws)

    def save(self, name: str, path: Union[str, Path] = Path.cwd()):
        path = Path(path)
        operator_path = path / name
        if operator_path.exists():
            raise FileExistsError(operator_path)
        operator_path.mkdir(parents=True)
        with open('{}/{}.yaml'.format(operator_path, name), 'w', encoding='utf-8') as f:
            info = yaml.safe_load(self._pipeline.pipeline.graph_repr.ir)
            info['name'] = name
            f.write(yaml.safe_dump(info))

    def push_to_hub(self, version: str = 'main'):
        # TODO: push to hub with new hub tool
        pass


def image_embedding_pipeline(model: Union[str, List[str]] = None, ensemble: str = None, name: str = None, version: str = None):
    pipe = None
    if name is not None:
        pipe = pipeline(name, tag=version)
        return pipe
    if model is not None:
        return ImageEmbeddingPipeline(model=model, ensemble=ensemble)
