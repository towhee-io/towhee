# coding : UTF-8
import sys
sys.path.append("../../tests")
import towhee
import time
import threading
import numpy as np
import random
import operator
from towhee import pipeline
from common import common_func as cf
from towhee.functional import DataCollection

class TestDataCollectionAPIsInvalid:
    """ Test case of invalid data collection interface """

    def test_data_collection_API_no_parameter(self, API_name):
        """
        target: test data collection APIs for invalid scenario
        method:  call APIs with no parameters
        expected: raise exception
        """
        API = eval("DataCollection.%s" % API_name)
        try:
            dc = API()
        except Exception as e:
            assert "missing" in str(e)

        return True

    def test_data_collection_API_not_supported_type(self, API_name):
        """
        target: test data collection APIs for invalid scenario
        method: call APIs with not supported data types
        expected: raise exception
        """
        if API_name in ["filter", "batch", "rolling", "pmap"]:
            return True
        not_support_datas = ["string", {1, "s", 2, 3}]
        API = eval("DataCollection.%s" % API_name)
        for not_support_data in not_support_datas:
            try:
                dc = API(not_support_data)
            except Exception as e:
                assert "no attribute" in str(e)

        return True

class TestDataCollectionAPIsValid:
    """ Test case of invalid data collection interface """

    def test_data_collection_stream_empty(self):
        """
        target: test stream() API for empty list
        method: create a streamed data collection from empty list
        expected: return stream successfully
        """
        data = []
        dc = DataCollection.stream(data)
        result = dc.is_stream
        assert result is True
        assert operator.eq(data, dc.to_list())

        return True

    def test_data_collection_stream_list(self):
        """
        target: test stream() API of DataCollection
        method: create a streamed data collection from list
        expected: return stream successfully
        """
        size = 5
        data = [random.random() for _ in range(size)]
        dc = DataCollection.stream(data)
        result = dc.is_stream
        assert result is True
        assert operator.eq(data, dc.to_list())

        return True

    def test_data_collection_stream_iter(self):
        """
        target: test stream() API of DataCollection
        method: create a streamed data collection from iter
        expected: return stream successfully
        """
        size = 5
        data = iter(range(size))
        dc = DataCollection.stream(data)
        result = dc.is_stream
        assert result is True

        return True

    def test_data_collection_stream_large_size(self):
        """
        target: test stream() API of DataCollection for large size data
        method: create a streamed data collection from long list
        expected: return stream successfully
        """
        size = 10000000
        data = [random.random() for _ in range(size)]
        dc = DataCollection.stream(data)
        result = dc.is_stream
        assert result is True
        assert operator.eq(data, dc.to_list())

        return True



