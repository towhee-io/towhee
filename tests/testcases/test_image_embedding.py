# coding : UTF-8
import sys
sys.path.append("../../tests")
import time
import threading
import numpy as np 
from towhee import AutoPipes, AutoConfig
from common import common_func as cf

data_path = "images/"
pipeline_name = 'text_image_embedding'


class TestImageEmbeddingInvalid:
    """ Test case of invalid text_image_embedding interface """

    def test_embedding_config_no_parameter(self):
        """
        target: test embedding for invalid scenario
        method:  embedding with no config parameter
        expected: raise exception
        """
        try:
            AutoConfig.load_config()
        except TypeError as e:
            print("Raise Exception: missing 1 required positional argument: 'name'")
            assert 'missing' in str(e)
        return True

    def test_embedding_pipeline_no_parameter(self):
        """
        target: test embedding for invalid scenario
        method:  embedding with no pipeline parameter
        expected: raise exception
        """
        try:
            AutoPipes.pipeline()
        except TypeError as e:
            print("Raise Exception: missing 1 required positional argument: 'name'")
            assert 'missing' in str(e)
        return True

    def test_embedding_wrong_pipeline_name(self):
        """
        target: test embedding for invalid scenario
        method:  embedding with wrong pipeline name
        expected: raise exception
        """
        try:
            AutoConfig.load_config([0])
        except TypeError as e:
            print("Raise Exception: unhashable type")
            assert 'unhashable type' in str(e)
        return True

    def test_embedding_nonexistent_pipeline_name(self):
        """
        target: test embedding for invalid scenario
        method:  embedding with nonexistent pipeline name
        expected: raise exception
        """
        try:
            AutoConfig.load_config(' ')
        except TypeError as e:
            print("Raise Exception: Can not find config")
        return True

    def test_embedding_nonexistent_model(self):
        """
        target: test embedding for invalid scenario
        method:  embedding with nonexistent pipeline model
        expected: raise exception
        """
        emb_conf = AutoConfig.load_config(pipeline_name)
        try:
            emb_conf.model = 'clip1000'
            AutoPipes.pipeline(pipeline_name, emb_conf)
        except RuntimeError as e:
            assert 'failed' in str(e)
        except KeyError as k:
            assert 'clip1000' in str(k)
        return True

    def test_embedding_nonexistent_image(self):
        """
        target: test embedding for invalid scenario
        method:  embedding with nonexistent image
        expected: raise exception
        """
        emb_pipe = AutoPipes.pipeline(pipeline_name)
        not_exist_img = "towhee_no_exist.jpg"
        try:
            emb_pipe(not_exist_img).to_list()
        except RuntimeError as e:
            print("Raise Exception: Read image towhee_no_exist.jpg failed")
            assert "failed" in str(e)
        return True

    def test_embedding_image_list(self):
        """
        target: test embedding with image list
        method:  embedding with image list
        expected: raise exception
        """
        N = 3
        img_list = []
        img = cf.create_image()
        for i in range(N):
            img_list.append(img)
        emb_pipe = AutoPipes.pipeline(pipeline_name)
        try:
            emb_pipe(img_list).to_list()
        except AttributeError as e:
            assert "list" in str(e)
        except RuntimeError as r:
            assert "list" in str(r)
        return True


class TestImageEmbeddingValid:
    """ Test case of valid embedding interface """

    def test_embedding_one_image(self, pipeline_model, embedding_size):
        """
        target: test embedding for normal case
        method:  embedding with one image
        expected: return embeddings
        """
        emb_conf = AutoConfig.load_config(pipeline_name)
        emb_conf.model = pipeline_model
        emb_pipe = AutoPipes.pipeline(pipeline_name, emb_conf)
        res = emb_pipe(data_path + "towhee_test_image0.jpg").to_list()[0][0].tolist()
        assert len(res) == embedding_size
        return True

    def test_embedding_same_images(self, pipeline_model, embedding_size):
        """
        target: test embedding for normal case
        method:  embedding with same images
        expected: return identical embeddings
        """
        emb_conf = AutoConfig.load_config(pipeline_name)
        emb_conf.model = pipeline_model
        emb_pipe = AutoPipes.pipeline(pipeline_name, emb_conf)
        res1 = emb_pipe(data_path + "towhee_test_image0.jpg").to_list()[0][0].tolist()
        res2 = emb_pipe(data_path + "towhee_test_image2.jpg").to_list()[0][0].tolist()
        assert len(res1) == embedding_size
        assert res1 == res2
        return True

    def test_embedding_different_images(self, pipeline_model, embedding_size):
        """
        target: test embedding for normal case
        method:  embedding with different images
        expected: return not identical embeddings
        """
        nums = 2
        embeddings = []
        emb_conf = AutoConfig.load_config(pipeline_name)
        emb_conf.model = pipeline_model
        emb_pipe = AutoPipes.pipeline(pipeline_name, emb_conf)
        for i in range(nums):
            embedding = emb_pipe(data_path + "towhee_test_image%d.jpg" % i).to_list()[0][0].tolist()
            embeddings.append(embedding)
            assert len(embedding) == embedding_size
        assert embeddings[0] is not embeddings[1]
        return True

    def test_embedding_one_image_multiple_times(self, pipeline_model, embedding_size):
        """
        target: test embedding for normal case
        method: embedding one image for multiple times
        expected: return identical embeddings
        """
        nums = 10
        emb_conf = AutoConfig.load_config(pipeline_name)
        emb_conf.model = pipeline_model
        emb_pipe = AutoPipes.pipeline(pipeline_name, emb_conf)
        embedding_last = emb_pipe(data_path + "towhee_test_image1.jpg").to_list()[0][0].tolist()
        for i in range(nums):
            embedding = emb_pipe(data_path + "towhee_test_image1.jpg").to_list()[0][0].tolist()
            assert len(embedding) == embedding_size
            assert embedding == embedding_last
            print("embedding images for %d round" % (i+1))
        return True

    def test_embedding_images_multiple_times(self, pipeline_model, embedding_size):
        """
        target: test embedding for normal case
        method:  embedding images for multiple times
        expected: return embeddings
        """
        nums = 2
        times = 10
        emb_conf = AutoConfig.load_config(pipeline_name)
        emb_conf.model = pipeline_model
        emb_pipe = AutoPipes.pipeline(pipeline_name, emb_conf)
        for i in range(times):
            embeddings = []
            for j in range(nums):
                embedding = emb_pipe(data_path + "towhee_test_image%d.jpg" % j).to_list()[0][0].tolist()
                embeddings.append(embedding)
                assert len(embedding) == embedding_size
            assert embeddings[0] is not embeddings[1]
            print("embedding images for %d round" % (i+1))

        return True

    def test_embedding_concurrent_multi_threads(self, pipeline_model, embedding_size):
        """
        target: test embedding for normal case
        method:  embedding with concurrent multi-processes
        expected: return embeddings
        """
        threads_num = 10
        threads = []
        emb_conf = AutoConfig.load_config(pipeline_name)
        emb_conf.model = pipeline_model
        emb_pipe = AutoPipes.pipeline(pipeline_name, emb_conf)

        def embedding():
            embeddings = emb_pipe(data_path + "towhee_test_image2.jpg").to_list()[0][0].tolist()
            assert len(embeddings) == embedding_size

        for i in range(threads_num):
            t = threading.Thread(target=embedding)
            threads.append(t)
            t.start()
            time.sleep(0.2)
        for t in threads:
            t.join()
        return True


class TestImageEmbeddingStress:
    """ Test case of stress """

    def test_embedding_more_times(self, pipeline_model, embedding_size):
        """
        target: test embedding for stress scenario
        method: embedding for N times
        expected: return embeddings
        """
        emb_conf = AutoConfig.load_config(pipeline_name)
        emb_conf.model = pipeline_model
        emb_pipe = AutoPipes.pipeline(pipeline_name, emb_conf)
        nums = 1000
        for i in range(nums):
            try:
                embedding = emb_pipe(data_path + "towhee_test_image0.jpg").to_list()[0][0].tolist()
                assert len(embedding) == embedding_size
            except Exception as e:
                print("Raise Exception: %s" % e)
            print("embedding images for %d round" % (i+1))
        return True


class TestImageEmbeddingPerformance:
    """ Test case of performance """

    def test_embedding_avg_time(self, pipeline_model, embedding_size):
        """
        target: test embedding for performance scenario
        method: embedding N times and calculate the average embedding time 
        expected: return embeddings
        """
        emb_conf = AutoConfig.load_config(pipeline_name)
        emb_conf.model = pipeline_model
        emb_pipe = AutoPipes.pipeline(pipeline_name, emb_conf)
        time_cost = []
        num = 10
        for i in range(num):
            try:
                time_start = time.time()
                embedding = emb_pipe(data_path + "towhee_test_image0.jpg").to_list()[0][0].tolist()
                time_cost.append(time.time() - time_start)
                assert len(embedding) == embedding_size
            except Exception as e:
                print("Raise Exception: %s" % e)
            print("embedding images for %d round" % (i+1))
        time_cost = np.array(time_cost)
        total_time = np.sum(time_cost)
        print(f"The total time is", total_time)
        avg_time = round(total_time/num, 3)
        print(f"The average time is", avg_time)
        
        return True
