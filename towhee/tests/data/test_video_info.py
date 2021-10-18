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


import unittest
import requests
import os

from towhee.data.video_info import get_video_info


class VideoInfoTest(unittest.TestCase):
    url = 'https://dl.fbaipublicfiles.com/pytorchvideo/projects/theatre.webm'
    filename = url.split('/', maxsplit=1)[-1]
    root = os.path.dirname(os.path.abspath(__file__))
    file_path = root + '/' + filename
    r = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(r.content)
        f.close()

    def test_video_info(self):
        info = get_video_info(self.file_path)
        self.assertEqual(info.fps, 30)
        self.assertEqual(info.width, 1280)
        self.assertEqual(info.height, 720)
        self.assertEqual(info.frame_count, 7128)
        self.assertEqual(info.duration, 237.6)


if __name__ == '__main__':
    unittest.main()
