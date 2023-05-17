# coding : UTF-8
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


from towhee import pipeline


class TestPipelineInvalid:
    """ Test case of invalid pipeline interface """

    def test_pipeline_no_params(self):
        """
        target: test pipeline for invalid scenario
        method:  call pipeline with no params
        expected: raise exception
        """
        try:
            embedding_pipeline = pipeline()
        except TypeError as e:
            print("Raise Exception: %s" % e)

    def test_pipeline_wrong_params(self):
        """
        target: test pipeline for invalid scenario
        method:  call pipeline with wrong pipeline name
        expected: raise exception
        """
        wrong_pipeline = "wrong-embedding"
        try:
            embedding_pipeline = pipeline(wrong_pipeline)
        except Exception as e:
            print("Raise Exception: %s" % e)


class TestPipelineValid:
    """ Test case of valid pipeline interface """

    def test_pipeline(self, pipeline_name):
        """
        target: test pipeline for image normal case
        method:  call pipeline with right pipeline name
        expected: return object
        """
        embedding_pipeline = pipeline(pipeline_name)
        assert "_pipeline" in dir(embedding_pipeline)





