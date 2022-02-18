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

from pathlib import Path
from towhee.hparam.hyperparameter import param_scope
from towhee.pipelines.alias_resolvers import get_resolver
from towhee.pipelines.base import PipelineBase
from towhee import Inject, pipeline
from towhee.utils.yaml_utils import load_yaml, dump_yaml


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
            info = load_yaml(self._pipeline.pipeline.graph_repr.ir)
            info['name'] = name
            dump_yaml(data = info, stream = f)

    def push_to_hub(self, version: str = 'main'):
        # TODO: push to hub with new hub tool
        pass


def image_embedding_pipeline(model: Union[str, List[str]] = None, ensemble: str = None, name: str = None, version: str = None):
    """Create a pipeline for image embedding tasks.

    An image embedding pipeline converts input images into feature vectors (embedding),
    which can be adapted to various vision tasks,
    such as image retrieval, image classifications, etc.

    There are two ways to instantiate an image embedding pipeline:

    1 - If `model` is passed to `image_embedding_pipeline`,
    a new pipeline will be generated for evaluation and benchmarking.

    ```python
    >>> pipe = image_embedding_pipeline(model='resnet101')
    >>> embedding = pipe('uri_to_image')
    ```

    The pipeline can be saved to file if the evaluation results seems good,
    and if you want to reuse this pipeline:
    ```python
    >>> pipe.save(name='my_image_embedding_pipeline', path='my_pipelines')
    ```

    You can also publish this pipeline to towhee hub to share it with the community.
    ```shell
    $ cd ${WORK_DIR}/my_pipelines/my_image_embedding_pipeline
    $ towhee publish # see towhee publish user guide from the terminal
    $ git commit && git push
    ```

    2 - Load a saved/shared pipeline from towhee hub:
    ```python
    >>> pipe = image_embedding_pipeline(name='your_name/my_image_embedding_pipeline')
    ```

    Args:
        model (Union[str, List[str]], optional): Backbone models for extracting image embedding.
            If there are more than one models, the model outputs will be fused with the `ensemble` model.
            Defaults to None.
        ensemble (str, optional): Ensemble model used to fuse backbone model outputs. Defaults to None.
        name (str, optional): Pipeline name. Defaults to None.
        version (str, optional): Version of the pipeline. Defaults to None.

    Returns:
        Pipeline: An image embedding pipeline.
    """
    pipe = None
    if name is not None:
        pipe = pipeline(name, tag=version)
        return pipe
    if model is not None:
        return ImageEmbeddingPipeline(model=model, ensemble=ensemble)
