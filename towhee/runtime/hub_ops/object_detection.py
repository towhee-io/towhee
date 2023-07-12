# Copyright 2023 Zilliz. All rights reserved.
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

from towhee.runtime.factory import HubOp


class ObjectDetection:
    """
    `Object detection <https://towhee.io/tasks/detail/operator?field_name=Computer-Vision&task_name=Object-Detection>`_
    is a computer vision technique that locates and identifies people, items, or other objects in an image. Object detection has applications
    in many areas of computer vision, including image retrieval, image annotation, vehicle counting, object tracking, etc.
    """

    yolov5: HubOp = HubOp('object_detection.yolov5')
    """
    Object Detection is a computer vision technique that locates and identifies people, items, or other objects in an image.
    Object detection has applications in many areas of computer vision, including image retrieval, image annotation,
    vehicle counting, object tracking, etc.
    This operator uses `PyTorch.yolov5 <https://pytorch.org/hub/ultralytics_yolov5/>`_ to detect the object.

    __init__(self)

    __call__(self, img: numpy.ndarray) -> Tuple[boxes, classes, scores]
        img(`ndarray`):
            Image data in ndarray format.
        Returns:
            boxes(`List[List[(int, int, int, int)]]`)
                A list of bounding boxes, Each bounding box is represented by the top-left and the bottom right points, i.e. (x1, y1, x2, y2).
            classes(`List[str]`):
                A list of prediction labels
            scores(`List[float]`):
                A list of confidence scores.

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        p = (
            pipe.input('path')
                .map('path', 'img', ops.image_decode())
                .map('img', ('box', 'class', 'score'), ops.object_detection.yolov5())
                .map(('img', 'box'), 'object', ops.image_crop(clamp=True))
                .output('img', 'object', 'class')
        )

        DataCollection(p('./test.png')).show()
    """

    yolov8: HubOp = HubOp('object_detection.yolov8')
    """
    __init__(self, model_name: str)
        model_name(`str`)
            Available models: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt

    __call__(self, img: numpy.ndarray) -> Tuple[boxes, classes, scores]
        img(`ndarray`):
            Image data in ndarray format.
        Returns:
            boxes(`List[List[(int, int, int, int)]]`)
                A list of bounding boxes, Each bounding box is represented by the top-left and the bottom right points, i.e. (x1, y1, x2, y2).
            classes(`List[str]`):
                A list of prediction labels
            scores(`List[float]`):
                A list of confidence scores.

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        p = (
            pipe.input('path')
                .map('path', 'img', ops.image_decode())
                .map('img', ('box', 'class', 'score'), ops.object_detection.yolov8())
                .map(('img', 'box'), 'object', ops.image_crop(clamp=True))
                .output('img', 'object', 'class')
        )

        DataCollection(p('./test.png')).show()
    """

    detectron2: HubOp = HubOp('object_detection.detectron2')
    """
    This operator uses Facebook's `Detectron2 <https://github.com/facebookresearch/detectron2>`_ library to
    compute bounding boxes, class labels, and class scores for detected objects in a given image.

    __init__(self, model_name: str = 'faster_rcnn_resnet50_c4', thresh: int = 0.5):
        model_name(`str`):
            A string indicating which model to use. Available options:faster_rcnn_resnet50_c4
            faster_rcnn_resnet50_dc5, faster_rcnn_resnet50_fpn, faster_rcnn_resnet101_c4,
            faster_rcnn_resnet101_dc5, faster_rcnn_resnet101_fpn, faster_rcnn_resnext101,
            retinanet_resnet50, retinanet_resnet101
        thresh(`float`):
            The threshold value for which an object is detected (default value: 0.5). Set
            this value lower to detect more objects at the expense of accuracy, or higher
            to reduce the total number of detections but increase the quality of detected objects.

    __call__(self, image: 'towhee.types.Image') -> Tuple[List]
        image(`towhe..types.Image`):
            Image data wrapped in a (as a Towhee Image).
        Returns:
            boxes(`List[List[(int, int, int, int)]]`)
                A list of bounding boxes, Each bounding box is represented by the top-left and the bottom right points, i.e. (x1, y1, x2, y2).
            classes(`List[str]`):
                A list of prediction labels
            scores(`List[float]`):
                A list of confidence scores.

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        p = (
            pipe.input('path')
                .map('path', 'img', ops.image_decode())
                .map('img', ('boxes', 'classes', 'scores'), ops.object_detection.detectron2(model_name='retinanet_resnet50'))
                .output('img', 'boxes', 'classes', 'scores')
        )

        DataCollection(p('./example.jpg')).show()
    """

    def __call__(self, *args, **kwargs):
        """
        Resolve the conflict issue that may be caused by ops users omitting the towhee namespace during use.
        """
        return HubOp('towhee.object_detection')(*args, **kwargs)

