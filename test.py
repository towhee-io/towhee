import time
from towhee.functional import DataCollection
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

mat = np.random.rand(100, 10000)

def math_map(x):
    np.dot(x, mat)

def get_list():
    return [np.random.rand(10000, 100) for _ in range(10)]

d = get_list()
t1 = time.time()
DataCollection(d).pmap2(math_map, 8).to_list()
t2 = time.time()
print('pmap2_8_worker:', t2 - t1)
DataCollection(d).pmap2(math_map, 4).to_list()
t3 = time.time()
print('pmap2_4_worker:', t3 - t2)
DataCollection(d).pmap2(math_map, 2).to_list()
t4 = time.time()
print('pmap2_2_worker:', t4 - t3)


t1 = time.time()
DataCollection(d).pmap1(math_map, 8).to_list()
t2 = time.time()
print('pmap1_8_worker:', t2 - t1)
DataCollection(d).pmap1(math_map, 4).to_list()
t3 = time.time()
print('pmap1_4_worker:', t3 - t2)
DataCollection(d).pmap1(math_map, 2).to_list()
t4 = time.time()
print('pmap1_2_worker:', t4 - t3)


t1 = time.time()
DataCollection(d).pmap(math_map, 8).to_list()
t2 = time.time()
print('pmap_8_worker:', t2 - t1)
DataCollection(d).pmap(math_map, 4).to_list()
t3 = time.time()
print('pmap_4_worker:', t3 - t2)
DataCollection(d).pmap(math_map, 2).to_list()
t4 = time.time()
print('pmap_2_worker:', t4 - t3)
