# coding : UTF-8
from operator import methodcaller
from test_image_embedding import *
from test_pipeline import *
from test_image_object_embedding import *

def pipeline_register():

    pipeline_names = ["image-embedding", "towhee/img_object_embedding"]

    return pipeline_names

def pipeline_runner():

    invalid_pipeline_obj = TestPipelineInvalid()
    for func in dir(TestPipelineInvalid):
        if not func.startswith("__"):
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
                res = methodcaller(func, pipeline_name)(valid_pipeline_obj)
                if res == None:
                    print("%s:%s PASS" % (func, pipeline_name))
                else:
                    print("%s:%s FAIL" % (func, pipeline_name))
    return True

def image_class_pipeline_register():

    pipeline_names = ["image-embedding"]

    return pipeline_names

def image_class_pipeline_runner():

    pipeline_names = image_class_pipeline_register()
    for pipeline_name in pipeline_names:
        invalid_embedding_obj = TestImageEmbeddingInvalid()
        for func in dir(TestImageEmbeddingInvalid):
            if not func.startswith("__"):
                res = methodcaller(func, pipeline_name)(invalid_embedding_obj)
                if res == 1:
                    print("%s:%s PASS" % (func, pipeline_name))
                else:
                    print("%s:%s FAIL" % (func, pipeline_name))
        valid_embedding_obj = TestImageEmbeddingValid()
        for func in dir(TestImageEmbeddingValid):
            if not func.startswith("__"):
                res = methodcaller(func, pipeline_name)(valid_embedding_obj)
                if res == 1:
                    print("%s:%s PASS" % (func, pipeline_name))
                else:
                    print("%s:%s FAIL" % (func, pipeline_name))

    return True

def image_obj_class_pipeline_register():

    pipeline_names = ["towhee/img_object_embedding"]

    return pipeline_names

def image_obj_class_pipeline_runner():

    pipeline_names = image_obj_class_pipeline_register()
    for pipeline_name in pipeline_names:
        invalid_embedding_obj = TestImageObjEmbeddingInvalid()
        for func in dir(TestImageObjEmbeddingInvalid):
            if not func.startswith("__"):
                res = methodcaller(func, pipeline_name)(invalid_embedding_obj)
                if res == 1:
                    print("%s:%s PASS" % (func, pipeline_name))
                else:
                    print("%s:%s FAIL" % (func, pipeline_name))
        valid_embedding_obj = TestImageObjEmbeddingValid()
        for func in dir(TestImageObjEmbeddingValid):
            if not func.startswith("__"):
                res = methodcaller(func, pipeline_name)(valid_embedding_obj)
                if res == 1:
                    print("%s:%s PASS" % (func, pipeline_name))
                else:
                    print("%s:%s FAIL" % (func, pipeline_name))

    return True

def test_caller():

    pipeline_runner()
    image_class_pipeline_runner()
    image_obj_class_pipeline_runner()

    return True


if __name__ == '__main__':

    test_caller()






