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

from towhee.pipeline.base import PipelineBase


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

    def __init__(self, model: str, ensemble: str, full_name: str):
        pass

    def __call__(self, image_path: str):
        pass

    def save(self, name: str, version: str, path: str):
        pass

    def push_to_hub(self):
        pass


# def image_embedding_pipeline(model: str = None, ensemble: str = None, name: str = None, version: str = None):
#
#   if name != None:
#       if pipeline_exists(name):
#           full_name = name
#       else:
#           full_name = pipeline_full_name(get_user_id(), name, version))
#           if not pipeline_exists(full_name):
#               raise PipelineNotExist()
#       return ImageEmbeddingPipeline(full_name = full_name)
#
#   if model != None:
#       return ImageEmbeddingPipeline(model=model, ensemble=ensemble)
