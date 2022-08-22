import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy
import towhee
from towhee.functional import DataCollection
import time
import ray


def test_function():
    import time
    time.sleep(0.1)
    # state = numpy.random.RandomState(123132312)
    # state2 = numpy.random.RandomState(1112121)
    # a = state.random((400,400))
    # b = state2.random((400,400))
    # c = numpy.dot(a, b)
    # c = numpy.dot(a, b)
    return 0


def test_mmap_pmap_combo(backend, function, num_worker):
    dc = DataCollection.range(10).stream().set_parallel(num_worker=num_worker, backend=backend)
    a, b, anull, bnull, cnull = dc.mmap([lambda x: function() + x+1, lambda x: function() + x*2, lambda x: function() + x*2, lambda x: function() + x*2,  lambda x: function() + x*2])
    d = b.pmap(lambda x: function() + x+1)
    e = d.pmap(lambda x : function() + x+1)
    f, g = a.mmap([lambda x: function() +  x+1, lambda x: function() + x*2])
    h = f.zip(e)
    i = h.zip(g, anull, bnull, cnull)

    return i.to_list()


if __name__ == '__main__':
    loops = 5
    ray.init(address='auto')
    

    ######## MMAP and PMAP RAY vs THREAD
    print('Starting MMAP and PMAP...')
    for worker in range(2, 8):
        avg_thread = 0
        avg_ray = 0
        for x in range(loops):
            start = time.time()
            result1 = test_mmap_pmap_combo('thread', test_function1, num_worker=worker)
            end = time.time()
            time1 = end - start

            start = time.time()
            result2 = test_mmap_pmap_combo('ray', test_function2, num_worker=worker)
            end = time.time()
            time2 = end - start
            
            avg_thread += time1
            avg_ray += time2

        avg_thread = avg_thread/loops
        avg_ray = avg_ray/loops

        print('Worker:', worker, 'Loops:', loops, 'Backend: Thread', 'Avg Time: ', avg_thread)
        print('Worker:', worker, 'Loops:', loops, 'Backend: Ray', 'Avg Time: ', avg_ray)
        print('Dif:', str(avg_thread - avg_ray))
