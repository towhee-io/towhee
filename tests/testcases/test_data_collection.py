# coding : UTF-8
import sys
sys.path.append("../../tests")
import towhee
import time
import threading
import numpy as np
import random
import operator
from common import common_func as cf
from towhee import pipe, ops


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
        if API_name in ["filter", "batch", "rolling", "pmap", "map", "mmap"]:
            return True
        not_support_datas = ["string", {1, "s", 2, 3}]
        API = eval("DataCollection.%s" % API_name)
        for not_support_data in not_support_datas:
            try:
                dc = API(not_support_data)
            except Exception as e:
                assert "no attribute" in str(e)

        return True


class TestPipeAPIsInvalid:
    """ Test case of invalid pipe interface """

    def test_data_collection_output_schema_invalid(self):
        """
        target: test output() API for invalid scenario
        method: 1. Output node not callable  2. Output invalid type(list)
                3. Output invalid type(set)  4. Output invalid type(number)
        expected: raise exception
        """
        # 1. Output node not callable
        try:
            towhee.pipe.input('a').map('a', 'b', lambda x: x+1).output('b', 'c')
        except Exception as e:
            assert "not declared" in str(e)

        # 2. Output invalid type(list)
        try:
            towhee.pipe.input('a').map('a', 'b', lambda x: x + 1).output(['b'])
        except Exception as e:
            assert "schema must be string" in str(e)

        # 3. Output invalid type(set)
        try:
            towhee.pipe.input('a').map('a', 'b', lambda x: x + 1).output({'b'})
        except Exception as e:
            assert "schema must be string" in str(e)

        # 4. Output invalid type(number)
        try:
            towhee.pipe.input('a').map('a', 'b', lambda x: x + 1).output(0)
        except Exception as e:
            assert "schema must be string" in str(e)

        return True

    def test_data_collection_map_invalid(self):
        """
        target: test map() API for invalid scenario
        method: 1. Operator not exist   2. Link not exist
                3. Func not valid
        expected: raise exception
        """
        # 1. Operator not exist
        try:
            towhee.pipe.input('path').map('path', 'img', ops.towhee.image_1()).output('img')
        except RuntimeError as e:
            assert "failed" in str(e)

        # 2. Link not exist
        """
        t = towhee.pipe.input('path').map('path', 'img', ops.towhee.image_decode()).output('img')
        try:
            t('https://github.com/towhee-io/towhee/raw/main/towhee.png').get()
        except Exception as e:
            assert "image" in str(e)
        """

        # 3. Func not valid
        try:
            p = towhee.pipe.input('a').map('a', 'b', lambda x: x + y).output('b')
            p(10).get()
        except Exception as e:
            assert "not" in str(e)
        return True

    def test_data_collection_flat_map_invalid(self):
        """
        target: test flat_map() API for invalid scenario
        method: Func not valid
        expected: raise exception
        """
        try:
            p = towhee.pipe.input('a').flat_map('a', 'b', lambda x: x + y).output('b')
            p(10).get()
        except Exception as e:
            assert "not" in str(e)
        return True

    def test_data_collection_filter_invalid(self):
        """
        target: test filter() API for invalid scenario
        method: filter_columns not exist, func invalid
        expected: raise exception
        """
        try:
            towhee.pipe.input('a').filter('a', 'b', lambda x: x < 0).output('b')
        except Exception as e:
            assert "missing" in str(e)

        try:
            p = towhee.pipe.input('a').filter('a', 'b', 'a', lambda x: x + y).output('b')
            p(10).get()
        except Exception as e:
            assert "not defined" in str(e)

        return True

    def test_data_collection_window_invalid(self):
        """
        target: test window() API for invalid scenario
        method: 1. input/output_schema type invalid     2. Size type invalid(not int)
                3. Size value invalid(<0)               4. Step type invalid(not int)
                5. Step value invalid(<0)
        expected: raise exception
        """
        # 1. input_schema/output_schema type invalid
        try:
            towhee.pipe.input('n1', 'n2') \
                       .window(('n1', 'n'), ('s1', 's'), 2, 2, lambda x, y: (sum(x), sum(y))) \
                       .output('s1', 's2')
        except Exception as e:
            assert "not declared" in str(e)

        # 2. Size type invalid (not int)
        size_type = ['B', [23], {0}]
        for size in size_type:
            try:
                towhee.pipe.input('n1', 'n2') \
                      .window(('n1', 'n2'), ('s1', 's2'), size, 2, lambda x, y: (sum(x), sum(y))) \
                      .output('s1', 's2')
            except Exception as e:
                assert "not int" in str(e)

        # 3. Size value invalid (<0)
        # If size value > the length of input data, towhee compute with size = the length of input data
        try:
            towhee.pipe.input('n1', 'n2') \
                       .window(('n1', 'n2'), ('s1', 's2'), -10, 2, lambda x, y: (sum(x), sum(y))) \
                       .output('s1', 's2')
        except Exception as e:
            assert "<=0" in str(e)

        # 4. Step type invalid(not int)
        step_type = ['B', [23], {0}]
        for step in step_type:
            try:
                towhee.pipe.input('n1', 'n2') \
                    .window(('n1', 'n2'), ('s1', 's2'), 2, step, lambda x, y: (sum(x), sum(y))) \
                    .output('s1', 's2')
            except Exception as e:
                assert "not int" in str(e)

        # 5. Step value invalid(<0)
        try:
            towhee.pipe.input('n1', 'n2') \
                       .window(('n1', 'n2'), ('s1', 's2'), 2, -1, lambda x, y: (sum(x), sum(y))) \
                       .output('s1', 's2')
        except Exception as e:
            assert "<=0" in str(e)

        return True

    def test_data_collection_time_window_invalid(self):
        """
        target: test time_window() API for invalid scenario
        method: 1. timestamp_col not declared        2. timestamp_col type invalid (not str)
                3. Size/step type invalid            4. Size/step value invalid
        expected: raise exception
        """
        # 1. timestamp_col not declared
        try:
            towhee.pipe.input('d') \
                  .flat_map('d', ('n1', 't'), lambda x: ((a, c) for a, c in x)) \
                  .time_window('n1', 's1', 'a', 3, 1, lambda x: x) \
                  .output('s1')
        except Exception as e:
            assert "not declared" in str(e)

        # 2. timestamp_col type invalid (not str)
        try:
            towhee.pipe.input('d') \
                  .flat_map('d', ('n1', 't'), lambda x: ((a, c) for a, c in x)) \
                  .time_window('n1', 's1', 1000, 3, 1, lambda x: x) \
                  .output('s1')
        except Exception as e:
            assert "not iterable" in str(e)

        # 3. Size/step type invalid
        size_type = ['B', [23], {0}]
        for size in size_type:
            try:
                towhee.pipe.input('d') \
                      .flat_map('d', ('n1', 't'), lambda x: ((a, c) for a, c in x)) \
                      .time_window('n1', 's1', 't', size, 1, lambda x: x) \
                      .output('s1')
            except Exception as e:
                assert "not int" in str(e)
        step_type = ['B', [23], {0}]
        for step in step_type:
            try:
                towhee.pipe.input('d') \
                    .flat_map('d', ('n1', 't'), lambda x: ((a, c) for a, c in x)) \
                    .time_window('n1', 's1', 't', 3, step, lambda x: x) \
                    .output('s1')
            except Exception as e:
                assert "not int" in str(e)

        # 4. Size/step value invalid
        try:
            towhee.pipe.input('d') \
                  .flat_map('d', ('n1', 't'), lambda x: ((a, c) for a, c in x)) \
                  .time_window('n1', 's1', 't', -3, -1, lambda x: x) \
                  .output('s1')
        except Exception as e:
            assert "<=0" in str(e)

        return True

    def test_data_collection_window_all_invalid(self):
        """
        target: test window_all() API for invalid scenario
        method: input/output_schema type invalid
        expected: raise exception
        """
        try:
            towhee.pipe.input('n1', 'n2') \
                  .flat_map(('n1', 'n2'), ('n1', 'n2'), lambda x, y: [(a, b) for a, b in zip(x, y)]) \
                  .window_all(('n1', 'n2'), ('s1', 's'), lambda x, y: (sum(x), sum(y))) \
                  .output('s1', 's2')
        except Exception as e:
            assert "not declared" in str(e)

        return True

    def test_data_collection_concat_invalid(self):
        """
        target: test concat() API for invalid scenario
        method: 1. Concat pipe not existed
                2. Concat pipelines which have conflicting apis
                3. concat empty
        expected: raise exception
        """
        # 1. Concat pipe not existed
        try:
            pipe0 = towhee.pipe.input('a', 'b', 'c')
            pipe2 = pipe0.map(('b', 'c'), 'e', lambda x, y: x - y)
            pipe2.concat(pipe).output('d', 'e')
        except Exception as e:
            assert "the parameter of concat must be Pipeline" in str(e)

        # 3. Concat pipelines which have conflicting apis
        try:
            pipe0 = towhee.pipe.input('a')
            pipe1 = pipe0.map('a', 'b', lambda x: x + 1).output('b')
            pipe2 = pipe0.map('a', 'c', lambda x: x + 2).output('c')
            towhee.pipe.concat(pipe1, pipe2).output('b', 'c')
        except Exception as e:
            assert "has no attribute" in str(e)

        # 4. concat empty
        try:
            pipe4 = towhee.pipe.input('a').map('a', 'b', lambda x: x + 1)
            pipe4.concat().output('b')
        except Exception as e:
            assert "The parameter of concat cannot be None" in str(e)

        return True

    def test_data_collection_batch_invalid(self):
        """
        target: test batch() API for invalid scenario
        method: input invalid type(not list)
        expected: raise exception
        """
        p = pipe.input('a').map('a', 'b', lambda x: x + 1).output('b')

        # input 'int'
        try:
            p.batch(1)
        except Exception as e:
            assert "'int' object is not iterable" in str(e)

        # input 'str'
        try:
            p.batch('a')
        except Exception as e:
            assert 'can only concatenate str (not "int") to str' in str(e)

        # input 'dict'
        try:
            p.batch({'a': 1})
        except Exception as e:
            assert 'can only concatenate str (not "int") to str' in str(e)


class TestOldDataCollectionAPIsValid:
    """ Test case of valid data collection interface """

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
        dc = towhee.dc(data)
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
        dc = towhee.dc(data)
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
        dc = towhee.dc(data)
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
        dc = towhee.dc(data)
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
        dc1 = towhee.dc(data)
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
        dc1 = towhee.dc(data)
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
        dc = towhee.dc(data)
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
        dc = towhee.dc(data)
        size = 3
        result = [list(batch) for batch in dc.rolling(size, drop_tail=False)]
        assert len(result) == data_size

        return True
    
    def test_data_collection_flatten(self):
        """
        target: test flatten() API for DataCollection 
        method: create a data collection and flatten it
        expected: return flatten successfully
        """
        data = 10
        dc = DataCollection.range(data)
        size = 3
        res = dc.batch(size)
        result = res.flatten().to_list()
        assert len(result) == data

        return True

    def test_data_collection_flatten_large_size(self):
        """
        target: test flatten() API for DataCollection 
        method: create a data collection and flatten it with large size
        expected: return flatten successfully
        """
        data = 10000000
        dc = DataCollection.range(data)
        size = 1000000
        res = dc.batch(size)
        result = res.flatten().to_list()
        assert len(result) == data

        return True
    
    def test_data_collection_flatten_size_empty(self):
        """
        target: test flatten() API for DataCollection 
        method: create a data collection and flatten it with empty size
        expected: return flatten successfully
        """
        data = 10
        dc = DataCollection.range(data)
        size = []
        res = dc.batch(size)
        result = res.flatten().to_list()
        assert len(result) == data

        return True

    def test_data_collection_flatten_inter(self):
        """
        target: test flatten() API for DataCollection
        method: create an iter data collection and flatten it
        expected: return flatten successfully
        """
        data_size = 6
        data = iter(range(data_size))
        dc = towhee.dc(data)
        size = 3
        res = dc.batch(size)
        result = res.flatten().to_list()

        assert len(result) == data_size

        return True


class TestDataCollectionAPIsValid:
    """ Test case of valid data collection interface """

    def test_data_collection_input_schema(self):
        """
        target: test input() API for pipeline
        method: Create a new pipeline and input tuple
        expected: input successfully
        """
        p = towhee.pipe.input('a', 'b', 'c')

        return True

    def test_data_collection_output_schema(self):
        """
        target: test output() API for pipeline
        method: Close and preload an existing pipeline
        expected: output successfully
        """
        p = towhee.pipe.input('a').map('a', 'b', lambda x: x+1).output('b')
        assert p(100).get() == [101]

        # Output empty
        q = towhee.pipe.input('a').map('a', 'b', lambda x: x + 1).output()
        assert q(10).get() == None

        return True

    def test_data_collection_map(self):
        """
        target: test map() API for pipeline
        method: Get the input data, apply it to a function, then output.
        expected: output successfully
        """
        # run with lambda
        p = towhee.pipe.input('a', 'b').map(('a', 'b'), 'c', lambda x, y: x + y).output('c')
        assert p(30, 52).get() == [82]

        # run with callable
        def func(x, y):
            return x + y, x - y
        q = towhee.pipe.input('a', 'b').map(('a', 'b'), ('c', 'd'), func).output('c', 'd')
        assert q(100, 50).get() == [150, 50]
        def func1(x):
            y = 0
        q = towhee.pipe.input('a').map('a', 'b', func1).output('b')
        assert q(100).get() == None

        # run with the operator in hub
        t = towhee.pipe.input('path').map('path', 'img', ops.towhee.image_decode()).output('img')
        res = t('https://github.com/towhee-io/towhee/raw/main/towhee_logo.png').get()
        assert len(res) == 1

        return True

    def test_data_collection_flat_map(self):
        """
        target: test flat_map() API for pipeline
        method: Get the input data, apply it to a function, then output.
        expected: output successfully
        """
        p = towhee.pipe.input('a').flat_map('a', 'b', lambda x: [e+10 for e in x]).output('b')
        res = p([1, 2, 3])
        assert res.get() == [11]
        assert res.get() == [12]
        assert res.get() == [13]

        return True

    def test_data_collection_filter(self):
        """
        target: test filter() API for pipeline
        method: Get the input data, apply it to a function, check the function
                with filter_columns,then output.
        expected: output successfully
        """
        p = towhee.pipe.input('a').filter('a', 'a', 'a', lambda x: x > 10).output('a')
        assert p(1).get() is None
        assert p(11).get() == [11]

        # The order of filter() and map() can't be reversed here!
        q = towhee.pipe.input('a').filter('a', 'a', 'a', lambda x: x > 10)\
                                  .map('a', 'b', lambda x: x + 1).output('b')
        assert q(5).get() is None
        assert q(20).get() == [21]
        return True

    def test_data_collection_window(self):
        """
        target: test window() API for pipeline
        method: Get the input data, apply it to a function over windows of given size,then output.
        expected: output successfully
        """
        p = towhee.pipe.input('n1', 'n2')\
                       .flat_map(('n1', 'n2'), ('n1', 'n2'), lambda x, y: [(a, b) for a, b in zip(x, y)])\
                       .window(('n1', 'n2'), ('s1', 's2'), 2, 2, lambda x, y: (sum(x), sum(y)))\
                       .output('s1', 's2')
        res = p([1, 2, 3, 4], [2, 3, 4, 5])
        assert res.get() == [3, 5]
        assert res.get() == [7, 9]

        return True

    def test_data_collection_time_window(self):
        """
        target: test time_window() API for pipeline
        method: Get the input data, apply it to a function over time_windows of given size,then output.
        expected: output successfully
        """
        p = towhee.pipe.input('d')\
                       .flat_map('d', ('n1', 'n2', 't'), lambda x: ((a, b, c) for a, b, c in x))\
                       .time_window(('n1', 'n2'), ('s1', 's2'), 't', 3, 3, lambda x, y: (sum(x), sum(y)))\
                       .output('s1', 's2')
        res = p([(i, i+1, i * 1000) for i in range(11) if i < 3 or i > 7])
        assert res.get() == [3, 6]
        assert res.get() == [8, 9]
        assert res.get() == [19, 21]

        return True

    def test_data_collection_window_all(self):
        """
        target: test window_all() API for pipeline
        method: Get the input data, apply it to a function over a window of all elements,then output.
        expected: output successfully
        """
        p = towhee.pipe.input('n1', 'n2')\
                       .flat_map(('n1', 'n2'), ('n1', 'n2'), lambda x, y: [(a, b) for a, b in zip(x, y)])\
                       .window_all(('n1', 'n2'), ('s1', 's2'), lambda x, y: (sum(x), sum(y)))\
                       .output('s1', 's2')
        res = p([1, 2, 3, 4], [2, 3, 4, 5])
        assert res.get() == [10, 14]
        res = p([1, 2, 3, 4], [2, 3])
        assert res.get() == [3, 5]

        return True

    def test_data_collection_concat(self):
        """
        target: test window_all() API for pipeline
        method: Concat one or more pipelines to the existing pipeline and update all data from each pipeline.
        expected: output successfully
        """
        # concat 1 pipeline
        pipe0 = towhee.pipe.input('a', 'b', 'c')
        pipe1 = pipe0.map('a', 'd', lambda x: x + 1)
        pipe2 = pipe0.map(('b', 'c'), 'e', lambda x, y: x - y)
        pipe3 = pipe2.concat(pipe1).output('d', 'e')
        assert pipe3(1, 2, 3).get() == [2, -1]

        # concat >1 pipelines
        # if concat same such as concat(pipe1, pipe1, pipe1) ,execute as one.
        pipe6 = towhee.pipe.input('a')
        pipe7 = pipe6.map('a', 'b', lambda x: x + 1)
        pipe8 = pipe6.map('a', 'c', lambda x: x + 2)
        pipe9 = pipe8.filter('a', 'b', 'a', lambda x: x > 1)
        pipe10 = pipe9.concat(pipe7, pipe8).output('b', 'c')
        assert pipe10(10).get() == [11, 12]

        # concat repeatedly
        pipe0 = towhee.pipe.input('a')
        pipe1 = pipe0.map('a', 'b', lambda x: x + 1)
        pipe2 = pipe0.filter('a', 'b', 'a', lambda x: x > 0)
        pipe3 = pipe2.concat(pipe1).concat(pipe1).output('b')
        assert pipe3(10).get() == [11]

        # concat itself
        pipe0 = towhee.pipe.input('a')
        p = pipe0.concat(pipe0).map('a', 'c', lambda x: x + 1).output('c')
        assert p(10).get() == [11]

        return True

    def test_data_collection_batch(self):
        """
        target: test batch() API for pipeline
        method: input a list to Batch run the callable pipeline
        expected: output successfully
        """
        p = pipe.input('a').map('a', 'b', lambda x: x + 1).output('b')
        res = p.batch([1, 2, 3])
        assert [r.get() for r in res] == [[2], [3], [4]]

        def func(x, y):
            return x + y, x - y
        p = pipe.input('a', 'b').map(('a', 'b'), ('c', 'd'), func).output('c', 'd')
        res = p.batch([[1, 1], [100, 100]])
        assert [r.get() for r in res] == [[2, 0], [200, 0]]

        # input 'tuple'
        p = pipe.input('a').map('a', 'b', lambda x: x + 1).output('b')
        res = p.batch((2, 3))
        assert [r.get() for r in res] == [[3], [4]]
