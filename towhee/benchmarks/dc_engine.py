import time
from pathlib import Path

from towhee import DataCollection


CACHE_PATH = Path(__file__).parent.resolve()
new_cache_path = CACHE_PATH / 'cache'

def three_dots(size, loops, threads):
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

def image_embedding(image, loops, threads):
    paths = [image] * loops
    time1 = time.time()
    dc = DataCollection(paths).stream().set_parallel(num_worker=threads) \
        .image_decode.cv2() \
        .image_embedding.timm(model_name = 'resnet50') \
        .to_list()
    time2 = time.time()
    del dc
    runtime = time2 - time1
    return runtime

def text_embedding(text, loops, threads):
    text = [text] * loops
    time1 = time.time()
    dc = DataCollection(text).stream().set_parallel(num_worker=threads) \
        .towhee.transformers_nlp_auto(model_name= 'albert-base-v1') \
        .to_list()
    time2 = time.time()
    del dc
    runtime = time2 - time1
    return runtime
