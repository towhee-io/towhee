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
import sys
import towhee


class TestCommandValid:
    """Test case of valid towhee command"""
    def test_command_lsS3_relative_path(self):
        """
        target: test command 'lsS3' relative path
        method: use lsS3 to list files in S3 relative path without bucket_name
        expected: get correct files
        """
        res = os.popen("towhee lsS3 -p test/test.pt").read()
        assert "test.pt" in res
        return True

    def test_command_downloadS3_relative_path(self):
        """
        target: test command 'downloadS3' relative path
        method: use downloadS3 to download files in S3 relative path without bucket_name
        expected: download successfully
        """
        if os.path.exists("./test.pt"):
            os.system("rm test.pt")
        res = os.popen("towhee downloadS3 -pl ./ -pb test/test.pt").read()
        assert "success" in res
        if os.path.exists("./test.pt"):
            os.system("rm test.pt")
            return True
        else:
            return False

    def test_command_package(self):
        """
        target: command 'package'
        method: pack a folder and checkout the generated file
        expected: pack successfully
        """
        if os.path.exists("/home/super/mpvit"):
            os.system("rm -rf mpvit")
        res = os.system("git clone https://towhee.io/image-embedding/mpvit.git")
        if res != 0:
            return False
        else:
            os.system("cd /home/super/mpvit && towhee package -n towhee -p mpvit")
            if os.path.exists("/home/super/mpvit/dist"):
                os.system("rm -rf /home/super/mpvit")
                return True
            else:
                return False


class TestCommandInvalid:
    """Test case of invalid towhee command"""
    def test_command_lsS3_nonexistent_path(self):
        """
        target: test command 'lsS3' nonexistent path
        method: use lsS3 to list files with nonexistent path
        expected: get error message
        """
        res = os.popen("towhee lsS3 -p abc/lpl").read()
        assert "not exist" in res
        return True

    def test_command_lsS3_absolute_path(self):
        """
        target: test command 'lsS3' absolute path
        method: use lsS3 to list files in S3 absolute path with bucket_name
        expected: get error message
        """
        res = os.popen("towhee lsS3 -p s3://pretrainedweights.towhee.io/test/test.pt").read()
        assert "not exist" in res
        return True

    def test_command_downloadS3_nonexistent_path(self):
        """
        target: test command 'downloadS3' nonexistent path
        method: use downloadS3 to download files with nonexistent path
        expected: get error message
        """
        res = os.popen("towhee downloadS3 -pl ./ -pb bac/lpl.pt").read()
        assert "not exist" in res
        return True

    def test_command_downloadS3_absolute_path(self):
        """
        target: test command 'downloadS3' absolute path
        method: use downloadS3 to download files in S3 absolute path with bucket_name
        expected: get error message
        """
        res = os.popen("towhee downloadS3 -pl ./ -pb s3://pretrainedweights.towhee.io/test/test.pt").read()
        assert "not exist" in res
        return True
