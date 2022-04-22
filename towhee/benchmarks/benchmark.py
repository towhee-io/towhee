#pylint: disable=wrong-import-position
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy #pylint: disable=unused-import
from pathlib import Path
import time

from towhee.hub.file_manager import FileManagerConfig, FileManager
from towhee.benchmarks import runner_engine, dc_engine


def startup():
    fmc = FileManagerConfig()
    cache_path = Path(__file__).parent.resolve()
    new_cache_path = cache_path / 'cache'
    pipeline_cache = cache_path / 'yamls'
    fmc.update_default_cache(new_cache_path)
    pipelines = list(pipeline_cache.rglob('*.yaml'))
    fmc.cache_local_pipeline(pipelines)
    FileManager(fmc)
    time.sleep(1)

def three_dots(loops, size, threads):
    x = runner_engine.three_dots(size, loops, threads)
    y = dc_engine.three_dots(size, loops, threads)
    print('Three Dots:')
    print('Runner:', x)
    print('DC', y)

def image_embedding(loops, threads):
    image =  str(Path(__file__).parent.resolve().parents[1] / 'towhee_logo.png')
    x = runner_engine.image_embedding(image, loops, threads)
    y = dc_engine.image_embedding(image, loops, threads)
    print('Image Embedding:')
    print('Runner:', x)
    print('DC', y)

startup()
# three_dots(100, 1000, 2)
image_embedding(50, 1)

