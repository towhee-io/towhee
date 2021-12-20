# coding : UTF-8
import sys
sys.path.append("../../tests")
import time
import threading
from towhee import pipeline

data_path = "../data_set/object_image/"

embedding_size = 1000

class TestImageObjEmbeddingInvalid:
    """ Test case of invalid embedding interface """

    def test_embedding_no_image(self, pipeline_name):
        """
        target: test embedding for invalid scenario
        method:  embedding with no image, now issue 154
        expected: raise exception
        """
        embedding_pipeline = pipeline(pipeline_name)
        try:
            embedding = embedding_pipeline()
        except RuntimeError as e:
            assert "Input data is empty" in str(e)

        return True

    def test_embedding_no_obj_image(self, pipeline_name):
        """
        target: test embedding for no object image
        method:  embedding with no object image
        expected: raise exception for empty result
        """
        img = "../../towhee_logo.png"
        embedding_pipeline = pipeline(pipeline_name)
        try:
            embedding = embedding_pipeline(img)
        except RuntimeError as e:
            assert "Unkown error" in str(e)

        return True

class TestImageObjEmbeddingValid:
    """ Test case of valid embedding interface """

    def test_embedding_one_image_single_obj(self, pipeline_name):
        """
        target: test embedding for normal case
        method:  embedding with one image
        expected: return embeddings
        """
        embedding_pipeline = pipeline(pipeline_name)
        embedding = embedding_pipeline(data_path + "towhee_test_image0.jpg")
        assert embedding[0][0].size == embedding_size

        return True

    def test_embedding_one_image_multiple_obj(self, pipeline_name):
        """
        target: test embedding for normal case
        method:  embedding with one image
        expected: return embeddings
        """
        embedding_pipeline = pipeline(pipeline_name)
        embedding = embedding_pipeline(data_path + "towhee_test_image1.jpg")
        assert embedding[0][0].size == embedding_size
        assert embedding[1][0].size == embedding_size
        assert len(embedding) == 2

        return True

    def test_embedding_same_images(self, pipeline_name):
        """
        target: test embedding for normal case
        method:  embedding with same images
        expected: return identical embeddings
        """
        embedding_pipeline = pipeline(pipeline_name)
        embedding = embedding_pipeline(data_path + "towhee_test_image0.jpg")
        embedding1 = embedding_pipeline(data_path + "towhee_test_image2.jpg")
        assert embedding[0][0].size == embedding_size
        assert embedding[0][0].all() == embedding1[0][0].all()

        return True

    def test_embedding_different_images(self, pipeline_name):
        """
        target: test embedding for normal case
        method:  embedding with different images
        expected: return not identical embeddings
        """
        nums = 2
        embeddings = []
        embedding_pipeline = pipeline(pipeline_name)
        for i in range(nums):
            embedding = embedding_pipeline(data_path + "towhee_test_image%i.jpg" % i)
            embeddings.append(embedding)
            assert embedding[0][0].size == embedding_size

        assert embeddings[0][0] is not embeddings[1][0]

        return True

    def test_embedding_one_image_multiple_times(self, pipeline_name):
        """
        target: test embedding for normal case
        method: embedding one image for multiple times
        expected: return identical embeddings
        """
        nums = 10
        embedding_pipeline = pipeline(pipeline_name)
        embedding_last = embedding_pipeline(data_path + "towhee_test_image0.jpg")
        for _ in range(nums):
            embedding = embedding_pipeline(data_path + "towhee_test_image0.jpg")
            assert embedding[0][0].size == embedding_size
        assert embedding[0][0].all() == embedding_last[0][0].all()

        return True

    def test_embedding_images_multiple_times(self, pipeline_name):
        """
        target: test embedding for normal case
        method:  embedding images for multiple times
        expected: return embeddings
        """
        nums = 2
        times = 10
        embedding_pipeline = pipeline(pipeline_name)
        for _ in range(times):
            embeddings = []
            for i in range(nums):
                embedding = embedding_pipeline(data_path + "towhee_test_image0.jpg")
                embeddings.append(embedding)
                assert embedding[0][0].size == embedding_size
            assert embeddings[0][0] is not embeddings[1][0]

        return True

    def test_embedding_concurrent_multi_threads(self, pipeline_name):
        """
        target: test embedding for normal case
        method:  embedding with concurrent multi-processes
        expected: return embeddings
        """
        threads_num = 10
        threads = []

        def embedding():
            embedding_pipeline = pipeline(pipeline_name)
            embedding = embedding_pipeline(data_path + "towhee_test_image0.jpg")
            assert embedding[0][0].size == embedding_size

        for i in range(threads_num):
            t = threading.Thread(target=embedding)
            threads.append(t)
            t.start()
            time.sleep(0.2)
        for t in threads:
            t.join()

        return True
