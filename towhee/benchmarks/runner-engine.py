import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
import numpy
import towhee
from towhee.hub.file_manager import FileManagerConfig, FileManager
from pathlib import Path

fmc = FileManagerConfig()
CACHE_PATH = Path(__file__).parent.resolve()
new_cache_path = CACHE_PATH / 'yaml'
pipeline_cache = CACHE_PATH / 'yamls'
fmc.update_default_cache(new_cache_path)
pipelines = list(pipeline_cache.rglob('*.yaml'))
fmc.cache_local_pipeline(pipelines)
FileManager(fmc)

pipeline = towhee.pipeline('filip-halt/numpy-benchmark')
y = []
for x in range(100):
    y.append(pipeline(x)[0][0])
end2 = time.time()
