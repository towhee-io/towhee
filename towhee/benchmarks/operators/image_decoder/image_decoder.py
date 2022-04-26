# encoding=utf-8
# pylint: skip-file
from collections import namedtuple

import cv2
import requests
import numpy as np

from towhee.operator.base import Operator, SharedType
from towhee.utils import ndarray_utils
from towhee.utils.log import engine_log


class ImageDecoder(Operator):
    def __init__(self):
        pass

    def _load_from_remote(image_url: str) -> np.ndarray:
        try:
            r = requests.get(image_url, timeout=(20, 20))
            if r.status_code // 100 != 2:
                engine_log.error('Download image from %s failed, error msg: %s, request code: %s ' % (image_url,
                                                                                                      r.text,
                                                                                                      r.status_code))
                return None
            arr = np.asarray(bytearray(r.content), dtype=np.uint8)
            return cv2.imdecode(arr, -1)
        except Exception as e:
            engine_log.error('Download image from %s failed, error msg: %s' % (image_url, str(e)))
            return False

    def _load_from_local(image_path: str) -> np.ndarray:
        return cv2.imread(image_path)

    def __call__(self, image_path: str):
        print(image_path)
        image_path = image_path[0] if isinstance(image_path, list) else image_path
        if image_path.startswith('http'):
            bgr_cv_image = ImageDecoder._load_from_remote(image_path)
        else:
            bgr_cv_image = ImageDecoder._load_from_local(image_path)
        if bgr_cv_image is None:
            err = 'Read image %s failed' % image_path
            engine_log.error(err)
            raise RuntimeError(err)

        rgb_cv_image = cv2.cvtColor(bgr_cv_image, cv2.COLOR_BGR2RGB)
        Result = namedtuple('Result', ['image'])
        return Result(image=ndarray_utils.from_ndarray(rgb_cv_image, 'RGB'))

    @property
    def shared_type(self):
        return SharedType.Shareable
