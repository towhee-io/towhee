# coding : UTF-8
import sys
sys.path.append("../../tests")
import time
import threading
import numpy as np 
from towhee import pipeline
from common import common_func as cf

data_path = "images/"

class TestImageEmbeddingInvalid:
    """ Test case of invalid embedding interface """

    def test_embedding_no_parameter(self, pipeline_name):
        """
        target: test embedding for invalid scenario
        method:  embedding with no image, now issue 154
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
        method:  embedding with no image, now issue 154
        expected: raise exception
        """
        embedding_pipeline = pipeline(pipeline_name)
        try:
            embedding = embedding_pipeline("")
        except Exception as e:
            print("Raise Exception: %s" % e)

        return True

    def test_embedding_not_exist_image(self, pipeline_name):
        """
        target: test embedding for invalid scenario
        method:  embedding with no image, now issue 154
        expected: raise exception
        """
        embedding_pipeline = pipeline(pipeline_name)
        not_exist_img = "towhee_no_exist.jpg"
        try:
            embedding = embedding_pipeline(not_exist_img)
        except Exception as e:
            print("Raise Exception: %s" % e)

        return True

    def test_embedding_image_list(self, pipeline_name):
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
        embedding_pipeline = pipeline(pipeline_name)
        try:
            embedding = embedding_pipeline(img_list)
        except Exception as e:
            print("Raise Exception: %s" % e)

        return True

class TestImageEmbeddingValid:
    """ Test case of valid embedding interface """

    def test_embedding_one_image(self, pipeline_name, embedding_size):
        """
        target: test embedding for normal case
        method:  embedding with one image
        expected: return embeddings
        """
        embedding_pipeline = pipeline(pipeline_name)
        embedding = embedding_pipeline(data_path + "towhee_test_image0.jpg")
        assert embedding.size == embedding_size

        return True

    def test_embedding_same_images(self, pipeline_name, embedding_size):
        """
        target: test embedding for normal case
        method:  embedding with same images
        expected: return identical embeddings
        """
        embedding_pipeline = pipeline(pipeline_name)
        embedding = embedding_pipeline(data_path + "towhee_test_image0.jpg")
        embedding1 = embedding_pipeline(data_path + "towhee_test_image2.jpg")
        assert embedding.size == embedding_size
        assert embedding.all() == embedding1.all()

        return True

    def test_embedding_different_images(self, pipeline_name, embedding_size):
        """
        target: test embedding for normal case
        method:  embedding with different images
        expected: return not identical embeddings
        """
        nums = 2
        embeddings = []
        embedding_pipeline = pipeline(pipeline_name)
        for i in range(nums):
            embedding = embedding_pipeline(data_path + "towhee_test_image%d.jpg" % i)
            embeddings.append(embedding)
            assert embedding.size == embedding_size

        assert embeddings[0] is not embeddings[1]

        return True

    def test_embedding_one_image_multiple_times(self, pipeline_name, embedding_size):
        """
        target: test embedding for normal case
        method: embedding one image for multiple times
        expected: return identical embeddings
        """
        nums = 10
        embedding_pipeline = pipeline(pipeline_name)
        embedding_last = embedding_pipeline(data_path + "towhee_test_image1.jpg")
        for i in range(nums):
            embedding = embedding_pipeline(data_path + "towhee_test_image1.jpg")
            assert embedding.size == embedding_size
            assert embedding.all() == embedding_last.all()
            print("embedding images for %d round" % (i+1))

        return True

    def test_embedding_images_multiple_times(self, pipeline_name, embedding_size):
        """
        target: test embedding for normal case
        method:  embedding images for multiple times
        expected: return embeddings
        """
        nums = 2
        times = 10
        embedding_pipeline = pipeline(pipeline_name)
        for i in range(times):
            embeddings = []
            for j in range(nums):
                embedding = embedding_pipeline(data_path + "towhee_test_image%d.jpg" % j)
                embeddings.append(embedding)
                assert embedding.size == embedding_size
            assert embeddings[0] is not embeddings[1]
            print("embedding images for %d round" % (i+1))

        return True

    def test_embedding_concurrent_multi_threads(self, pipeline_name, embedding_size):
        """
        target: test embedding for normal case
        method:  embedding with concurrent multi-processes
        expected: return embeddings
        """
        threads_num = 10
        threads = []

        def embedding():
            embedding_pipeline = pipeline(pipeline_name)
            embedding = embedding_pipeline(data_path + "towhee_test_image2.jpg")
            assert embedding.size == embedding_size

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

    def test_embedding_more_times(self,pipeline_name):
       """
        target: test embedding for stress scenario
        method: embedding for N times
        expected: return embeddings
       """
       embedding_pipeline = pipeline(pipeline_name)
       nums = 1000
       for i in range(nums):
            try:
                embedding = embedding_pipeline(data_path + "towhee_test_image0.jpg")
            except Exception as e:
                print("Raise Exception: %s" % e)

       print("embedding images for %d round" % (i+1))

       return True

class TestImageEmbeddingPerformance:
    """ Test case of performance """

    def test_embedding_avg_time(self, pipeline_name):
        """
        target: test embedding for performance scenario
        method: embedding N times and calculate the average embedding time 
        expected: return embeddings
        """
        embedding_pipeline = pipeline(pipeline_name)
        avg_time = 0
        time_cost = []
        num = 10
        for i in range (num):
            try:
                time_start = time.time()
                embedding = embedding_pipeline(data_path + "towhee_test_image0.jpg")
                time_cost.append(time.time() - time_start)
            except Exception as e:
                print( "Raise Exception: %s" % e)
        time_cost = np.array(time_cost)
        total_time = np.sum(time_cost)
        print(f"The total time is",total_time)
        avg_time = round(total_time/num, 3)
        print(f"The average time is",avg_time)

        
        return True
