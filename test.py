import time
from towhee.functional import DataCollection

t1 = time.time()
DataCollection([2,1,3,4,6]).pmap(lambda x: x+1, 5)
t2 = time.time()
print('time:', t2 - t1)
re = DataCollection([2,1,3,4,6]).pmap1(lambda x: x+1, 5)
t3 = time.time()
print('time:', t3 - t2)
re = DataCollection([2,1,3,4,6]).pmap1(lambda x: x+1, 5)
t4 = time.time()
print('time:', t4 - t3)
