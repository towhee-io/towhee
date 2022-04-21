import time
from pathlib import Path

from towhee import DataCollection


CACHE_PATH = Path(__file__).parent.resolve()
new_cache_path = CACHE_PATH / 'cache'

def three_dots(loops, size, threads):
    time1 = time.time()
    dc = DataCollection.range(loops).set_parallel(num_worker=threads) \
        .filip_halt.numpy_benchmark_operator(size = size) \
        .filip_halt.numpy_benchmark_operator(size = size) \
        .filip_halt.numpy_benchmark_operator(size = size) \
        .to_list()
    time2 = time.time()
    del dc
    runtime = time2 - time1

    return runtime
