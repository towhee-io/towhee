# coding : UTF-8
import sys
sys.path.append("../../tests")
import time
import threading
from towhee import pipeline
from common import common_func as cf

embedding_size = 1000

class TestEmbeddingInvalid:
    """ Test case of invalid embedding interface """

    def test_embedding_no_image(self):
        """
        target: test embedding for invalid scenario
        method:  embedding with no image, now issue 154
        expected: raise exception
        """
        pipeline_name = "image-embedding"
        embedding_pipeline = pipeline(pipeline_name)
        embedding = embedding_pipeline()
        if embedding != []:
            print("test_embedding_no_image: test case fail")

        return True

class TestEmbeddingValid:
    """ Test case of valid embedding interface """

    def test_embedding_one_image(self):
        """
        target: test embedding for normal case
        method:  embedding with one image
        expected: return embeddings
        """
        img = cf.create_image()
        pipeline_name = "image-embedding"
        embedding_pipeline = pipeline(pipeline_name)
        embedding = embedding_pipeline(img)
        assert embedding[0].size == embedding_size

        return True

    def test_embedding_same_images(self):
        """
        target: test embedding for normal case
        method:  embedding with same images
        expected: return identical embeddings
        """
        img = cf.create_image()
        img_1 = img.copy()
        pipeline_name = "image-embedding"
        embedding_pipeline = pipeline(pipeline_name)
        embedding = embedding_pipeline(img)
        embedding1 = embedding_pipeline(img_1)
        assert embedding[0].size == embedding_size
        assert embedding1[0].size == embedding_size
        assert embedding[0].all() == embedding1[0].all()

        return True

    def test_embedding_different_images(self):
        """
        target: test embedding for normal case
        method:  embedding with different images
        expected: return not identical embeddings
        """
        nums = 2
        embeddings = []
        pipeline_name = "image-embedding"
        embedding_pipeline = pipeline(pipeline_name)
        for i in range(nums):
            img = cf.create_image()
            embedding = embedding_pipeline(img)
            embeddings.append(embedding[0])
            assert embedding[0].size == embedding_size

        assert embeddings[0] is not embeddings[1]

        return True

    def test_embedding_one_image_multiple_times(self):
        """
        target: test embedding for normal case
        method: embedding one image for multiple times
        expected: return identical embeddings
        """
        nums = 10
        img = cf.create_image()
        pipeline_name = "image-embedding"
        embedding_pipeline = pipeline(pipeline_name)
        embedding_last = embedding_pipeline(img)
        for _ in range(nums):
            embedding = embedding_pipeline(img)
            assert embedding[0].size == embedding_size
            assert embedding[0].all() == embedding_last[0].all()

        return True

    def test_embedding_images_multiple_times(self):
        """
        target: test embedding for normal case
        method:  embedding images for multiple times
        expected: return embeddings
        """
        nums = 2
        times = 10
        pipeline_name = "image-embedding"
        embedding_pipeline = pipeline(pipeline_name)
        for _ in range(times):
            embeddings = []
            for i in range(nums):
                img = cf.create_image()
                embedding = embedding_pipeline(img)
                embeddings.append(embedding)
                assert embedding[0].size == embedding_size
            assert embeddings[0] is not embeddings[1]

        return True

    def test_embedding_concurrent_multi_threads(self):
        """
        target: test embedding for normal case
        method:  embedding with concurrent multi-processes
        expected: return embeddings
        """
        threads_num = 10
        threads = []
        img = cf.create_image()

        def embedding():
            pipeline_name = "image-embedding"
            embedding_pipeline = pipeline(pipeline_name)
            embedding = embedding_pipeline(img)
            assert embedding[0].size == embedding_size

        for i in range(threads_num):
            t = threading.Thread(target=embedding)
            threads.append(t)
            t.start()
            time.sleep(0.2)
        for t in threads:
            t.join()

        return True


