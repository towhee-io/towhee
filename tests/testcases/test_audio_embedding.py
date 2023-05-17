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


import numbers
import sys
sys.path.append("../../tests")
import time
import threading
import numpy as np 
from towhee import AutoPipes, AutoConfig
from common import common_func as cf


data_path = "audios/"
pipeline_name = 'video_embedding'


class TestAudioEmbeddingInvalid:
    """ Test case of invalid embedding interface """

    def test_embedding_no_parameter(self, pipeline_name):
        """
        target: test embedding for invalid scenario
        method: embedding with no audio
        expected: raise exception
        """
        embedding_pipeline = pipeline(pipeline_name)
        try:
            embedding = embedding_pipeline()
        except RuntimeError as e:
            print("Raise Exception: %s" % e)

        return True

    def test_embedding_wrong_parameter(self, pipeline_name):
        """
        target: test embedding for invalid scenario
        method: embedding with empty parameter
        expected: raise exception
        """
        embedding_pipeline = pipeline(pipeline_name)
        try:
            embedding = embedding_pipeline("")
        except Exception as e:
            print("Raise Exception: %s" % e)

        return True

    def test_embedding_not_exist_audio(self, pipeline_name):
        """
        target: test embedding for invalid scenario
        method:  embedding with not exist audio
        expected: raise exception
        """
        embedding_pipeline = pipeline(pipeline_name)
        not_exist_img = "towhee_no_exist.jpg"
        try:
            embedding = embedding_pipeline(not_exist_img)
        except Exception as e:
            print("Raise Exception: %s" % e)

        return True

    def test_embedding_audio_list(self, pipeline_name):
        """
        target: test embedding with image list
        method: embedding with audio list
        expected: raise exception
        """
        N = 3
        img_list = []
        img = cf.create_image()
        for i in range(N):
            img_list.append(img)
        embedding_pipeline = pipeline(pipeline_name)
        try:
            embedding = embedding_pipeline(img_list)
        except Exception as e:
            print("Raise Exception: %s" % e)

        return True


class TestAudioEmbeddingValid:
    """ Test case of valid embedding interface """

    def test_embedding_one_audio(self, pipeline_name, embedding_size):
        """
        target: test embedding for normal case
        method: embedding with one audio
        expected: return embeddings
        """
        embedding_pipeline = pipeline(pipeline_name)
        embedding = embedding_pipeline(data_path + "towhee_test_audio_0.wav")
        assert embedding[0][0][0].size == embedding_size

        return True

    def test_embedding_same_audios(self, pipeline_name, embedding_size):
        """
        target: test embedding for normal case
        method: embedding with same audios
        expected: return identical embeddings
        """
        embedding_pipeline = pipeline(pipeline_name)
        embedding = embedding_pipeline(data_path + "towhee_test_audio_0.wav")
        embedding1 = embedding_pipeline(data_path + "towhee_test_audio_2.wav")
        assert embedding[0][0][0].size == embedding_size
        assert embedding[0][0].all() == embedding1[0][0].all()

        return True

    def test_embedding_different_audios(self, pipeline_name, embedding_size):
        """
        target: test embedding for normal case
        method: embedding with different audios
        expected: return not identical embeddings
        """
        nums = 2
        embeddings = []
        embedding_pipeline = pipeline(pipeline_name)
        for i in range(nums):
            embedding = embedding_pipeline(data_path + "towhee_test_audio_%d.wav" % i)
            embeddings.append(embedding[0][0])
            assert embedding[0][0][0].size == embedding_size

        assert embeddings[0] is not embeddings[1]

        return True

    def test_embedding_one_audio_multiple_times(self, pipeline_name, embedding_size):
        """
        target: test embedding for normal case
        method: embedding one audio for multiple times
        expected: return identical embeddings
        """
        nums = 2
        embedding_pipeline = pipeline(pipeline_name)
        embedding_last = embedding_pipeline(data_path + "towhee_test_audio_0.wav")
        for i in range(nums):
            embedding = embedding_pipeline(data_path + "towhee_test_audio_0.wav")
            assert embedding[0][0][0].size == embedding_size
            assert embedding[0][0].all() == embedding_last[0][0].all()
            print("embedding audios for %d round" % (i+1))

        return True

    def test_embedding_audios_multiple_times(self, pipeline_name, embedding_size):
        """
        target: test embedding for normal case
        method: embedding audios for multiple times
        expected: return embeddings
        """
        nums = 2
        times = 2
        embedding_pipeline = pipeline(pipeline_name)
        for i in range(times):
            embeddings = []
            for j in range(nums):
                embedding = embedding_pipeline(data_path + "towhee_test_audio_%d.wav" % j)
                embeddings.append(embedding)
                assert embedding[0][0][0].size == embedding_size
            assert embeddings[0] is not embeddings[1]
            print("embedding audios for %d round" % (i+1))

        return True

    def test_embedding_concurrent_multi_threads(self, pipeline_name, embedding_size):
        """
        target: test embedding for normal case
        method: embedding with concurrent multi-processes
        expected: return embeddings
        """
        threads_num = 10
        threads = []

        def embedding():
            embedding_pipeline = pipeline(pipeline_name)
            embedding = embedding_pipeline(data_path + "towhee_test_audio_0.wav")
            assert embedding[0][0][0].size == embedding_size

        for i in range(threads_num):
            t = threading.Thread(target=embedding)
            threads.append(t)
            t.start()
            time.sleep(0.2)
        for t in threads:
            t.join()

        return True


class TestAudioEmbeddingStress:
    """ Test case of stress """

    def test_embedding_more_times(self, pipeline_name, embedding_size):
        """
        target: test embedding for stress scenario
        method: embedding for N times
        expected: return embeddings
        """
        nums = 1000
        for i in range(nums):
            embedding_pipeline = pipeline(pipeline_name)
            try:
                embedding = embedding_pipeline(data_path + "towhee_test_audio_1.wav")
                assert embedding[0][0][0].size == embedding_size
            except Exception as e:
                print("Raise Exception: %s" % e)
            print("embedding audios for %d round" % (i+1))

        return True


class TestAudioEmbeddingPerformance:
    """ Test case of performance """

    def test_embedding_avg_time(self, pipeline_name, embedding_size):
        """
        target: test embedding for performance scenario
        method: embedding N times and calculate the average embedding time 
        expected: return embeddings
        """
        embedding_pipeline = pipeline(pipeline_name)
        avg_time = 0
        time_cost = []
        num = 10
        for i in range(num):
            try:
                time_start = time.time()
                embedding = embedding_pipeline(data_path + "towhee_test_audio_1.wav")
                time_cost.append(time.time() - time_start)
                assert embedding[0][0][0].size == embedding_size
            except Exception as e:
                print("Raise Exception: %s" % e)
            print("embedding audios for %d round" % (i+1))
        time_cost = np.array(time_cost)
        total_time = np.sum(time_cost)
        print(f"The total time is", total_time)
        avg_time = round(total_time/num, 3)
        print(f"The average time is", avg_time)
        
        return True


