# coding : UTF-8
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
            assert "pipeline() missing 1 required positional argument: 'task'" in str(e)

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
            assert "Incorrect pipeline format" in str(e)

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





