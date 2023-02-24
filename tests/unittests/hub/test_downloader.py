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
from pathlib import Path
from tempfile import TemporaryDirectory

from towhee.hub.downloader import _HubFiles, download_operator, operator_tag_path


class TestDownloader(unittest.TestCase):
    """
    Simple hub download and run test.
    """

    def test_hubfiles(self):
        meta = [
            {
                'path': 'README.md',
                'mode': '100644',
                'type': 'blob',
                'size': 5310,
                'sha': '133cd84e7a4bf39b49ffb18f3f2628afe93dd5b4',
                'downloadLink': 'https://towhee.io/api/v1/repos/image-embedding/isc/git/blobs/133cd84e7a4bf39b49ffb18f3f2628afe93dd5b4'
            },
            {
                'path': 'benchmark',
                'mode': '040000',
                'type': 'tree',
                'size': 0,
                'sha': '65c110ada6d3fb4e477cc57b67026e37a4499e39',
                'downloadLink': 'https://towhee.io/api/v1/repos/image-embedding/isc/git/trees/65c110ada6d3fb4e477cc57b67026e37a4499e39'
            },
            {
                'path': 'benchmark/qps_test.py',
                'mode': '100644',
                'type': 'blob',
                'size': 3617,
                'sha': '2bc4af18560f0597d5efd037aef7111aba01d140',
                'downloadLink': 'https://towhee.io/api/v1/repos/image-embedding/isc/git/blobs/2bc4af18560f0597d5efd037aef7111aba01d140'
            }
        ]
        with TemporaryDirectory(dir='./') as root:
            fs = _HubFiles(root, 'v1-0.1', meta)
            tag_path = fs.get_tag_path()
            self.assertEqual(tag_path, operator_tag_path(root, 'v1-0.1'))
            self.assertEqual(tag_path.absolute(), (Path(root) / 'versions' / 'v1_0_1').absolute())
            self.assertFalse(tag_path.is_dir())
            tag_path = fs.get_tag_path(True)
            self.assertTrue(tag_path.is_dir())
            self.assertEqual(tag_path.absolute(), (Path(root) / 'versions' / 'v1_0_1').absolute())
            self.assertEqual(fs.file_path, (Path(root) / 'files').absolute())
            self.assertTrue(fs.file_path.is_dir())
            self.assertIsNone(fs.requirements)

            meta.append(
                {
                    'path': 'requirements.txt',
                    'mode': '100644',
                    'type': 'blob',
                    'size': 3617,
                    'sha': '2bc4af18560f0597d5efd037aef7111aba01d140',
                    'downloadLink': 'https://towhee.io/api/v1/repos/image-embedding/isc/git/blobs/2bc4af18560f0597d5efd037aef7111aba01d140'
                }
            )
            self.assertEqual(fs.requirements.absolute(), (Path(root) / 'versions' / 'v1_0_1' / 'requirements.txt').absolute())
            fs.requirements.touch()
            self.assertIsNotNone(fs.requirements)
            self.assertEqual(len(fs.symlink_pair()), 3)
            self.assertEqual(len(fs.local_url_pair()), 3)


    def test_download_op(self):
        with TemporaryDirectory(dir='./') as root:
            download_operator('image-decode', 'cv2-rgb', 'main', root)
            op_root = Path(root)
            self.assertTrue((op_root / 'files').is_dir())
            self.assertTrue((op_root / 'versions' / 'main').is_dir())
