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
from towhee.command.s3 import S3Bucket

# pylint: disable=C0103
# pylint: disable=W0613
class MockClient(object):
    """
    Mock s3 client
    """
    def upload_file(self, file_upload, bucket_name, object_name, Config=None):
        return True

    def upload_part(self, Bucket, Key, PartNumber, UploadId, Body):
        return True

    def create_multipart_upload(self, Bucket, Key):
        mpu = {'UploadId':1}
        return mpu

    def head_object(self, Bucket, Key):
        return {'ETag': 'abcdefghijk'}

    def md5_compare(self, file_name, md5):
        return False

    def s3_md5sum(self, file):
        return None

    def download_file(self, bucket_name, object_name, file_name, Config=None):
        return True

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        return True

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=100):
        list_key = []
        list_key.append({'Key':'/test/test'})
        mock_list_content = {
            'Contents': list_key
        }
        return mock_list_content

class TestS3(unittest.TestCase):
    """
    Unittests for towhee cmdline.
    """
    def test_upload_file(self):
        bucket = S3Bucket()
        bucket.s3 = MockClient()
        self.assertEqual(bucket.upload_normal('./test_s3.py', 'test1'), True)

    def test_upload_files(self):
        bucket = S3Bucket()
        bucket.s3 = MockClient()
        self.assertEqual(bucket.upload_files('./', '/__w/towhee/towhee/tests/unittests/command/test_s3.py'), True)

    def test_ls(self):
        bucket = S3Bucket()
        bucket.s3 = MockClient()
        self.assertEqual(bucket.get_list_s3('./'), [])

    def test_download(self):
        bucket = S3Bucket()
        bucket.s3 = MockClient()
        self.assertEqual(bucket.download_file('test.txt', './'), True)

    def test_downloads(self):
        bucket = S3Bucket()
        bucket.s3 = MockClient()
        self.assertEqual(bucket.download_files('~/test', './'), True)

if __name__ == '__main__':
    unittest.main()
