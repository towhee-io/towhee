# coding : UTF-8
from operator import methodcaller
from test_image_embedding import *
from test_pipeline import *

if __name__ == '__main__':

    invalid_pipeline_obj = TestPipelineInvalid()
    for func in dir(TestPipelineInvalid):
        if not func.startswith("__"):
            res = methodcaller(func)(invalid_pipeline_obj)
            if res == None:
                print("%s PASS" % func)
            else:
                print("%s FAIL" % func)

    valid_pipeline_obj = TestPipelineValid()
    for func in dir(TestPipelineValid):
        if not func.startswith("__"):
            res = methodcaller(func)(valid_pipeline_obj)
            if res == None:
                print("%s PASS" % func)
            else:
                print("%s FAIL" % func)

    invalid_embedding_obj = TestEmbeddingInvalid()
    for func in dir(TestEmbeddingInvalid):
        if not func.startswith("__"):
            res = methodcaller(func)(invalid_embedding_obj)
            if res == 1:
                print("%s PASS" % func)
            else:
                print("%s FAIL" % func)

    valid_embedding_obj = TestEmbeddingValid()
    for func in dir(TestEmbeddingValid):
        if not func.startswith("__"):
            res = methodcaller(func)(valid_embedding_obj)
            if res == 1:
                print("%s PASS" % func)
            else:
                print("%s FAIL" % func)




