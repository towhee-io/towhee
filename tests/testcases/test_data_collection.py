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
    
    def test_data_collection_map_lambda(self):
        """
        target: test filter() API for DataCollection 
        method: create a data collection and map data with lambda
        expected: return map successfully
        """
        size = 5
        dc = DataCollection.range(size)
        dc = dc.map(lambda x: x+1)
        result = dc.to_list()
        assert len(result) == size

        return True

    def test_data_collection_map_inter(self):
        """
        target: test map() API for DataCollection 
        method: create an iter data collection and map it
        expected: return map successfully
        """
        size = 5
        data = iter(range(size))
        dc = DataCollection(data)
        dc = dc.map(str)
        result = dc.to_list()
        assert len(result) == size
        
        return True
     
    def test_data_collection_map_empty(self):
        """
        target: test map() API for DataCollection 
        method: create an empty data collection and map it
        expected: return map successfully
        """
        data = []
        dc = DataCollection(data)
        dc = dc.map(str)
        result = dc.to_list()
        assert result == data
        
        return True

    def test_data_collection_map_large_size(self):
        """
        target: test map() API for DataCollection 
        method: create a large size data collection and map it
        expected: return map successfully
        """
        data = 10000000
        dc = DataCollection.range(data)
        dc = dc.map(str)
        result = dc.to_list()
        assert len(result) == data
        
        return True
    
    def test_data_collection_filter_lambda(self):
        """
        target: test filter() API for DataCollection 
        method: create a data collection and filter data with lambda
        expected: return filter successfully
        """
        size = 5
        dc = DataCollection.range(size)
        dc = dc.filter(lambda x: x+1)
        result = dc.to_list()
        assert len(result) == size

        return True

    def test_data_collection_filter_inter(self):
        """
        target: test filter() API for DataCollection 
        method: create an iter data collection and filter it
        expected: return filter successfully
        """
        size = 5
        data = iter(range(size))
        dc = DataCollection(data)
        dc = dc.filter(str)
        result = dc.to_list()
        assert len(result) == size
        
        return True
     
    def test_data_collection_filter_empty(self):
        """
        target: test filter() API for DataCollection 
        method: create an empty data collection and filter it
        expected: return filter successfully
        """
        data = []
        dc = DataCollection(data)
        dc = dc.filter(str)
        result = dc.to_list()
        assert result == data

        return True

    def test_data_collection_filter_large_size(self):
        """
        target: test filter() API for DataCollection 
        method: create a large size data collection and filter it
        expected: return filter successfully
        """
        data = 10000000
        dc = DataCollection.range(data)
        dc = dc.filter(str)
        result = dc.to_list()
        assert len(result) == data

        return True
    
    def test_data_collection_zip_lambda(self):
        """
        target: test zip() API for DataCollection 
        method: create a data collection and zip  with lambda
        expected: return zip successfully
        """
        size = 5
        dc1 = DataCollection.range(size)
        dc2 = dc1.map(lambda x: x+1)
        dc3 = dc1.zip(dc2)
        result = dc3.to_list()
        assert len(result) == size
        return True

    def test_data_collection_zip_inter(self):
        """
        target: test zip() API for DataCollection 
        method: create an iter data collection and zip it
        expected: return zip successfully
        """
        size = 7
        data = iter(range(size))
        dc1 = DataCollection(data)
        dc2 = dc1.map(str)
        dc3 = dc1.zip(dc2)
        result = dc3.to_list()
        assert len(result) == int(size/2)
        
        return True
     
    def test_data_collection_zip_empty(self):
        """
        target: test zip() API for DataCollection 
        method: create an empty data collection and zip it
        expected: return zip successfully
        """
        data = []
        dc1 = DataCollection(data)
        size = 1
        dc2 = DataCollection.range(size)
        dc3 = dc1.zip(dc2)
        result = dc3.to_list()
        assert result == data

        return True

    def test_data_collection_zip_large_size(self):
        """
        target: test zip() API for DataCollection 
        method: create a large size data collection and zip it
        expected: return zip successfully
        """
        data = 10000000
        dc1 = DataCollection.range(data)
        dc2 = dc1.filter(str)
        dc3 = dc1.zip(dc2)
        result = dc3.to_list()
        assert len(result) == data

        return True

    def test_data_collection_batch_data(self):
        """
        target: test batch() API for DataCollection 
        method: create a data collection and batch it
        expected: return batch successfully
        """
        data = 10
        dc = DataCollection.range(data)
        size = 3
        result = [list(batch) for batch in dc.batch(size, drop_tail=True)]
        assert len(result) == int(data/size) 

        return True

    def test_data_collection_batch_inter(self):
        """
        target: test batch() API for DataCollection 
        method: create an iter data collection and batch it
        expected: return batch successfully
        """
        data_size = 6
        data = iter(range(data_size))
        dc = DataCollection(data)
        size = 3
        result = [list(batch) for batch in dc.batch(size, drop_tail=True)]
        assert len(result) == int(data_size/size) 

        return True

    def test_data_collection_batch_large_size(self):
        """
        target: test batch() API for DataCollection 
        method: create a large size data collection and batch with large size
        expected: return batch successfully
        """
        data = 10000000
        dc = DataCollection.range(data)
        size = 1000000
        result = [list(batch) for batch in dc.batch(size, drop_tail=True)]
        assert len(result) == int(data/size)   

        return True

    def test_data_collection_batch_size_empty(self):
        """
        target: test batch() API for DataCollection 
        method: create a data collection and batch size is empty
        expected: return batch successfully
        """
        data = 5
        dc = DataCollection.range(data)
        size = []
        result = [list(batch) for batch in dc.batch(size)]
        assert len(result) == 1

        return True
    
    
    def test_data_collection_rolling_drop_head(self):
        """ 
        target: test rolling() API for DataCollection 
        method: create a data collection and rolling it with drop_head is False
        expected: return rolling successfully
        """
        data = 5
        dc = DataCollection.range(data)
        size = 3
        result = [list(batch) for batch in dc.rolling(size, drop_head=False)]
        assert len(result) == data

        return True
    
    def test_data_collection_rolling_drop_tail(self):
        """ 
        target: test rolling() API for DataCollection 
        method: create a data collection and rolling it with drop_tail is False
        expected: return rolling successfully
        """
        data = 10
        dc = DataCollection.range(data)
        size = 3
        result = [list(batch) for batch in dc.rolling(size, drop_tail=False)]
        assert len(result) == data

        return True

    def test_data_collection_rolling_large_size(self):
        """    
        target: test rolling() API for DataCollection 
        method: create a large size data collection and rolling it with large size
        expected: return rolling successfully
        """
        data = 100000
        dc = DataCollection.range(data)
        size = 1000
        result = [list(batch) for batch in dc.rolling(size, drop_tail=False)]
        assert len(result) == data

        return True

    def test_data_collection_rolling_size_empty(self):
        """
        target: test rolling() API for DataCollection 
        method: create a data collection and rolling size is empty
        expected: return rolling successfully
        """
        data = 10
        dc = DataCollection.range(data)
        size = []
        result = [list(batch) for batch in dc.rolling(size)]
        assert len(result) == 0

        return True

    def test_data_collection_rolling_inter(self):
        """
        target: test rolling() API for DataCollection 
        method: create an iter data collection and rolling it
        expected: return rolling successfully
        """
        data_size = 6
        data = iter(range(data_size))
        dc = DataCollection(data)
        size = 3
        result = [list(batch) for batch in dc.rolling(size, drop_tail=False)]
        assert len(result) == data_size

        return True
    
    def test_data_collection_flaten(self):
        """
        target: test flaten() API for DataCollection 
        method: create a data collection and flaten it
        expected: return flaten successfully
        """
        data = 10
        dc = DataCollection.range(data)
        size = 3
        res = dc.batch(size)
        result = res.flaten().to_list()
        assert len(result) == data

        return True

    def test_data_collection_flaten_large_size(self):
        """
        target: test flaten() API for DataCollection 
        method: create a data collection and flaten it with large size
        expected: return flaten successfully
        """
        data = 10000000
        dc = DataCollection.range(data)
        size = 1000000
        res = dc.batch(size)
        result = res.flaten().to_list()
        assert len(result) == data

        return True
    
    def test_data_collection_flaten_size_empty(self):
        """
        target: test flaten() API for DataCollection 
        method: create a data collection and flaten it with empty size
        expected: return flaten successfully
        """
        data = 10
        dc = DataCollection.range(data)
        size = []
        res = dc.batch(size)
        result = res.flaten().to_list()
        assert len(result) == data

        return True

    def test_data_collection_flaten_inter(self):
        """
        target: test flaten() API for DataCollection
        method: create an iter data collection and flaten it
        expected: return flaten successfully
        """
        data_size = 6
        data = iter(range(data_size))
        dc = DataCollection(data)
        size = 3
        result = [list(batch) for batch in dc.rolling(size, drop_tail=False)]
        assert len(result) == data_size

        return True
