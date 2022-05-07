# coding : UTF-8
from operator import methodcaller
from test_image_embedding import *
from test_pipeline import *
from test_audio_embedding import *
from test_data_collection import *

def pipeline_register():

    pipeline_names = ["image-embedding", "towhee/image-embedding-efficientnetb5",
                      "towhee/image-embedding-efficientnetb7", "towhee/image-embedding-resnet101",
                      "towhee/image-embedding-swinbase", "towhee/image-embedding-swinlarge",
                      "towhee/image-embedding-vitlarge", "towhee/audio-embedding-clmr",
                      "towhee/audio-embedding-vggish"]

    return pipeline_names

def pipeline_runner():

    invalid_pipeline_obj = TestPipelineInvalid()
    for func in dir(TestPipelineInvalid):
        if not func.startswith("__"):
            print("Testing %s" % func)
            res = methodcaller(func)(invalid_pipeline_obj)
            if res == None:
                print("%s PASS" % func)
            else:
                print("%s FAIL" % func)

    pipeline_names = pipeline_register()
    for pipeline_name in pipeline_names:
        valid_pipeline_obj = TestPipelineValid()
        for func in dir(TestPipelineValid):
            if not func.startswith("__"):
                print("Testing %s:%s" % (func, pipeline_name))
                res = methodcaller(func, pipeline_name)(valid_pipeline_obj)
                if res == None:
                    print("%s:%s PASS" % (func, pipeline_name))
                else:
                    print("%s:%s FAIL" % (func, pipeline_name))
    return True

def image_class_pipeline_register():
    # skip efficientnetb7 image pipeline for memory shortage
    # pipeline_names = ["image-embedding", "towhee/image-embedding-efficientnetb5",
    #                   "towhee/image-embedding-efficientnetb7", "towhee/image-embedding-resnet101",
    #                   "towhee/image-embedding-resnet50", "towhee/image-embedding-swinbase",
    #                   "towhee/image-embedding-swinlarge", "towhee/image-embedding-vitlarge"]
    # embedding_sizes = [2048, 2048, 2560, 2048, 2048, 1024, 1536, 1024]

    pipeline_names = ["image-embedding", "towhee/image-embedding-efficientnetb5",
                      "towhee/image-embedding-resnet101", "towhee/image-embedding-resnet50",
                      "towhee/image-embedding-swinbase", "towhee/image-embedding-swinlarge",
                      "towhee/image-embedding-vitlarge"]

    embedding_sizes = [2048, 2048, 2048, 2048, 1024, 1536, 1024]

    # skip multiple threads tests for memory shortage
    skipped_cases = ["test_embedding_concurrent_multi_threads", "test_embedding_more_times", "test_embedding_avg_time"]
 

    return pipeline_names, embedding_sizes, skipped_cases

def image_class_pipeline_runner():

    pipeline_names, embedding_sizes, skipped_cases = image_class_pipeline_register()
    for (pipeline_name, embedding_size_each) in zip(pipeline_names, embedding_sizes):
        invalid_embedding_obj = TestImageEmbeddingInvalid()
        for func in dir(TestImageEmbeddingInvalid):
            if func in skipped_cases:
                continue
            if not func.startswith("__"):
                print("Testing %s:%s" % (func, pipeline_name))
                res = methodcaller(func, pipeline_name)(invalid_embedding_obj)
                if res == 1:
                    print("%s:%s PASS" % (func, pipeline_name))
                else:
                    print("%s:%s FAIL" % (func, pipeline_name))
        valid_embedding_obj = TestImageEmbeddingValid()
        for func in dir(TestImageEmbeddingValid):
            if func in skipped_cases:
                continue
            if not func.startswith("__"):
                print("Testing %s:%s" % (func, pipeline_name))
                res = methodcaller(func, pipeline_name, embedding_size_each)(valid_embedding_obj)
                if res == 1:
                    print("%s:%s PASS" % (func, pipeline_name))
                else:
                    print("%s:%s FAIL" % (func, pipeline_name))
        
        test_valid_embedding = TestImageEmbeddingStress()
        for func in dir(TestImageEmbeddingStress):
            if func in skipped_cases:
                continue
            if not func.startswith("__"):
                print("Testing %s:%s" % (func, pipeline_name))
                res = methodcaller(func, pipeline_name, embedding_size_each)(test_valid_embedding)
                if res == 1:
                    print("%s:%s PASS" % (func, pipeline_name))
                else:
                    print("%s:%s FAIL" % (func, pipeline_name))

        test_valid_embedding_per = TestImageEmbeddingPerformance()
        for func in dir(TestImageEmbeddingPerformance):
            if func in skipped_cases:
                continue
            if not func.startswith("__"):
                print("Testing %s:%s" % (func, pipeline_name))
                res = methodcaller(func, pipeline_name, embedding_size_each)(test_valid_embedding_per)
                if res == 1:
                    print("%s:%s PASS" % (func, pipeline_name))
                else:
                    print("%s:%s FAIL" % (func, pipeline_name))

    return True

def audio_class_pipeline_register():

    # skip clmr audio pipeline for memory shortage
    # pipeline_names = ["towhee/audio-embedding-clmr", "towhee/audio-embedding-vggish"]
    # embedding_sizes = [512, 128]
    pipeline_names = ["towhee/audio-embedding-vggish"]

    embedding_sizes = [128]

    # skip multiple threads tests for memory shortage
    skipped_cases = ["test_embedding_concurrent_multi_threads", "test_embedding_more_times", "test_embedding_avg_time"]


    return pipeline_names, embedding_sizes, skipped_cases

def audio_class_pipeline_runner():

    pipeline_names, embedding_sizes, skipped_cases = audio_class_pipeline_register()
    for (pipeline_name, embedding_size_each) in zip(pipeline_names, embedding_sizes):
        invalid_embedding_obj = TestAudioEmbeddingInvalid()
        for func in dir(TestAudioEmbeddingInvalid):
            if func in skipped_cases:
                continue
            if not func.startswith("__"):
                print("Testing %s:%s" % (func, pipeline_name))
                res = methodcaller(func, pipeline_name)(invalid_embedding_obj)
                if res == 1:
                    print("%s:%s PASS" % (func, pipeline_name))
                else:
                    print("%s:%s FAIL" % (func, pipeline_name))
        valid_embedding_obj = TestAudioEmbeddingValid()
        for func in dir(TestAudioEmbeddingValid):
            if func in skipped_cases:
                continue
            if not func.startswith("__"):
                print("Testing %s:%s" % (func, pipeline_name))
                res = methodcaller(func, pipeline_name, embedding_size_each)(valid_embedding_obj)
                if res == 1:
                    print("%s:%s PASS" % (func, pipeline_name))
                else:
                    print("%s:%s FAIL" % (func, pipeline_name))

        test_valid_embedding = TestAudioEmbeddingStress()
        for func in dir(TestAudioEmbeddingStress):
            if func in skipped_cases:
                continue
            if not func.startswith("__"):
                print("Testing %s:%s" % (func, pipeline_name))
                res = methodcaller(func, pipeline_name, embedding_size_each)(test_valid_embedding)
                if res == 1:
                    print("%s:%s PASS" % (func, pipeline_name))
                else:
                    print("%s:%s FAIL" % (func, pipeline_name))     

        test_valid_embedding_per = TestAudioEmbeddingPerformance()
        for func in dir(TestAudioEmbeddingPerformance):
            if func in skipped_cases:
                continue
            if not func.startswith("__"):
                print("Testing %s:%s" % (func, pipeline_name))
                res = methodcaller(func, pipeline_name, embedding_size_each)(test_valid_embedding_per)
                if res == 1:
                    print("%s:%s PASS" % (func, pipeline_name))
                else:
                    print("%s:%s FAIL" % (func, pipeline_name))    

    return True

def data_collection_API_register():

    data_colection_API_names = ["stream", "unstream", "map", "filter", "zip", "batch", "rolling", 
                                #"flaten",
                                "exception_safe", "safe", "fill_empty", "drop_empty", "pmap", "mmap", "set_parallel"]

    # skip multiple threads tests for memory shortage
    skipped_API = []

    return data_colection_API_names, skipped_API


def data_collection_API_cases_runner():

    data_colection_API_names, skipped_API = data_collection_API_register()
    for data_colection_API_name in data_colection_API_names:
        invalid_data_collection_API_obj = TestDataCollectionAPIsInvalid()
        for func in dir(invalid_data_collection_API_obj):
            if func in skipped_API:
                continue
            if not func.startswith("__"):
                print("Testing %s:%s" % (func, data_colection_API_name))
                res = methodcaller(func, data_colection_API_name)(invalid_data_collection_API_obj)
                if res == 1:
                    print("%s:%s PASS" % (func, data_colection_API_name))
                else:
                    print("%s:%s FAIL" % (func, data_colection_API_name))

    data_collection_API_obj = TestDataCollectionAPIsValid()
    for func in dir(data_collection_API_obj):
        if func in skipped_API:
            continue
        if not func.startswith("__"):
            print("Testing %s" % func)
            res = methodcaller(func)(data_collection_API_obj)
            if res == 1:
                print("%s PASS" % func)
            else:
                print("%s FAIL" % func)

def test_caller():

    pipeline_runner()
    # image_class_pipeline_runner()
    data_collection_API_cases_runner()
    # skip audio tests for issue 463
    # audio_class_pipeline_runner()

    return True


if __name__ == '__main__':

    test_caller()






