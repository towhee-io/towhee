import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy #pylint: disable=unused-import, wrong-import-position
from pathlib import Path #pylint: disable=wrong-import-position

from towhee.hub.file_manager import FileManagerConfig, FileManager #pylint: disable=wrong-import-position
from towhee.benchmarks import runner_engine, dc_engine #pylint: disable=wrong-import-position

def startup():
    fmc = FileManagerConfig()
    cache_path = Path(__file__).parent.resolve()
    new_cache_path = cache_path / 'cache'
    pipeline_cache = cache_path / 'yamls'
    fmc.update_default_cache(new_cache_path)
    pipelines = list(pipeline_cache.rglob('*.yaml'))
    fmc.cache_local_pipeline(pipelines)
    FileManager(fmc)

def three_dots(loops, size, threads):
    x = runner_engine.three_dots(loops, size, threads)
    y = dc_engine.three_dots(loops, size, threads*3) # Runner is threads per task, dc is threads per pipeline
    print('Three Dots:')
    print('Runner:', x)
    print('DC', y)

startup()
three_dots(100, 1000, 2)

