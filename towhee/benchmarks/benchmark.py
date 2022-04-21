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


fmc = FileManagerConfig()
CACHE_PATH = Path(__file__).parent.resolve()
new_cache_path = CACHE_PATH / 'cache'
pipeline_cache = CACHE_PATH / 'yamls'
fmc.update_default_cache(new_cache_path)
pipelines = list(pipeline_cache.rglob('*.yaml'))
fmc.cache_local_pipeline(pipelines)
FileManager(fmc)

x = runner_engine.three_dots(1000, 500, 6)
print(x)
y = dc_engine.three_dots(1000, 500, 18)
print(y)


