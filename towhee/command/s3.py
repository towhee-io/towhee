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

from towhee.utils.thirdparty.boto3_utils import boto3
import re
import os
import math
import hashlib
from configparser import ConfigParser, NoSectionError
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig

def get_conf() -> dict:
    cf = ConfigParser()
    real_path = os.path.expanduser('~/.aws/credentials')
    cf.read(real_path)
    try:
        access_id = cf.get('default', 'aws_access_key_id')
        access_key = cf.get('default', 'aws_secret_access_key')
    except NoSectionError:
        access_id = ''
        access_key = ''
    s3_conf = {
        'ACCESS_KEY': access_id,
        'SECRET_KEY': access_key,
        'BUCKET_NAME': 'pretrainedweights.towhee.io',
    }
    return s3_conf

class S3Bucket(object):
    """
    S3 bucket upload ls download functions
    """
    def __init__(self):
        s3_conf = get_conf()
        self.access_key = s3_conf.get('ACCESS_KEY')
        self.secret_key = s3_conf.get('SECRET_KEY')
        self.bucket_name = s3_conf.get('BUCKET_NAME')

        # connect s3
        if self.access_key == '' or self.secret_key == '':
            self.s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        else:
            self.s3 = boto3.client(
                service_name='s3',
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                )

    def upload_normal(self, path_prefix, file_upload):
        """
        upload file.
        """
        gb = 1024 ** 3
        #default config
        config = TransferConfig(multipart_threshold=5*gb, max_concurrency=10, use_threads=True)
        file_name = os.path.basename(file_upload)
        object_name = os.path.join(path_prefix,file_name)
        try:
            self.s3.upload_file(file_upload, self.bucket_name, object_name, Config=config)
        except ClientError as e:

            print('error happend!' + str(e))
            return False
        print('upload done!')
        return True

    def upload_files(self, path_bucket, path_local):
        """
        upload big file.
        args:
            path_bucket (`str`):
                upload path in bucket
            path_local (`str`):
                real local path
        """
        # multipart upload
        real_path = os.path.abspath(path_local)
        chunk_size = 52428800
        source_size = os.stat(real_path).st_size
        print('source_size=', source_size)
        chunk_count = int(math.ceil(source_size/float(chunk_size)))
        mpu = self.s3.create_multipart_upload(Bucket=self.bucket_name, Key=path_bucket)
        part_info = {'Parts': []}
        with open(real_path, 'rb') as fp:
            for i in range(chunk_count):
                offset = chunk_size * i
                remain_bytes = min(chunk_size, source_size-offset)
                data = fp.read(remain_bytes)
                md5s = hashlib.md5(data)
                new_etag = '"%s"' % md5s.hexdigest()
                try:
                    self.s3.upload_part(Bucket=self.bucket_name,Key=path_bucket, PartNumber=i+1,
                                    UploadId=mpu['UploadId'],Body=data)
                except Exception as exc:  # pylint: disable=W0703
                    print('error occurred.', exc)
                    return False
                print('uploading %s %s'%(real_path, str(i/chunk_count)))
                parts={
                    'PartNumber': i+1,
                    'ETag':new_etag
                }
                part_info['Parts'].append(parts)
        print('%s uploaded!' % (real_path))
        self.s3.complete_multipart_upload(Bucket=self.bucket_name,Key=path_bucket,
                                          UploadId=mpu['UploadId'],
                                          MultipartUpload=part_info)
        print('%s uploaded success!' % (real_path))
        return True

    def download_file(self, object_name, path_local):
        """
        download the single file from s3 to local dir
        """
        gb = 1024**3
        config = TransferConfig(multipart_threshold=2*gb, max_concurrency=10, use_threads=True)
        suffix = object_name.split('.')[-1]
        if path_local[-len(suffix):] == suffix:
            file_name = path_local
            dir_name = os.path.dirname(file_name)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
        else:
            if not os.path.exists(path_local):
                os.mkdir(path_local)
            file_name = os.path.join(path_local, os.path.basename(object_name))
        print(object_name, file_name)
        print(self.s3_md5sum(object_name))
        if not self.md5_compare(file_name, self.s3_md5sum(object_name)):
            try:
                self.s3.download_file(self.bucket_name, object_name, file_name, Config=config)
            except Exception as exc:  # pylint: disable=W0703
                print('download single file error occurred.', exc)
                return False
            print('download ok', object_name)
        else:
            print('files in local folder.')
        return True

    def download_files(self, path_prefix, path_local):
        """
        download files from s3
        """
        gb = 1024**3
        config = TransferConfig(multipart_threshold=2*gb, max_concurrency=10, use_threads=True)
        try:
            list_content = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=path_prefix)['Contents']
        except KeyError:
            print('path %s is not exist. please check path[name_space]/[operator_name]/[version]/[file]'%path_prefix)
            return False
        for key in list_content:
            name = os.path.basename(key['Key'])
            object_name = key['Key']
            local_file = path_local+name
            if not self.md5_compare(local_file, self.s3_md5sum(object_name)):
                if not os.path.exists(path_local):
                    os.makedirs(path_local)
                file_name = os.path.join(path_local, name)
                try:
                    self.s3.download_file(self.bucket_name, object_name, file_name, Config=config)
                except Exception as exc:  # pylint: disable=W0703
                    print('download files error occurred.', exc)
                    return False
                print('download files %s in %s success!'%(name, path_local))
            else:
                print('files in local folder.')
        return True

    def get_list_s3(self, obj_floder_path):
        """
        list items in this path
        args:
            obj_floder_path (`str`):
                the path
        returns:
            files (`List[str]`):
                list of files' name in this path
        """
        file_list = []
        response = self.s3.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=obj_floder_path,
            MaxKeys=1000,
           )
        try:
            for file in response['Contents']:
                s = str(file['Key'])
                p = re.compile(r'.*/(.*)(\..*)')
                if p.search(s):
                    s1 = p.search(s).group(1)
                    s2 = p.search(s).group(2)
                    result = s1 + s2
                    file_list.append(result)
        except KeyError:
            print('path %s is not exist. please check path[name_space]/[operator_name]/[version]/[file]'%obj_floder_path)
        return file_list

    def s3_md5sum(self, resource_name):
        try:
            md5sum = self.s3.head_object(
                Bucket=self.bucket_name,
                Key=resource_name
            )['ETag'][1:-1]
        except ClientError:
            md5sum = None
            pass
        return md5sum

    def check_md5(self, file_path):
        try:
            md5obj = hashlib.md5()
            with open(file_path, 'rb') as fd:
                for data in iter(lambda: fd.read(50 * 1024 * 1024), b''):
                    md5obj.update(data)
            local_md5 = md5obj.hexdigest()
        except FileNotFoundError:
            local_md5 = None
        return local_md5

    def etag_checksum(self, file_path, chunk_size=50 * 1024 * 1024):
        """
        ETag check docs:
            https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity.html
        example:
            https://stackoverflow.com/questions/26415923/boto-get-md5-s3-file
        """
        try:
            md5s = []
            with open(file_path, 'rb') as f:
                for data in iter(lambda: f.read(chunk_size), b''):
                    md5s.append(hashlib.md5(data).digest())
            m = hashlib.md5(b''.join(md5s))
            res = '{}-{}'.format(m.hexdigest(), len(md5s))
        except FileNotFoundError:
            res = None
        return res

    def md5_compare(self, file_path, s3_file_md5):
        if s3_file_md5 is None:
            return False
        if '-' in s3_file_md5 and s3_file_md5 == self.etag_checksum(file_path):
            return True
        if '-' not in s3_file_md5 and s3_file_md5 == self.check_md5(file_path):
            return True
        return False
