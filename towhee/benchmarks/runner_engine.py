
import time
from pathlib import Path
import yaml

import towhee
from towhee.utils.yaml_utils import load_yaml

CACHE_PATH = Path(__file__).parent.resolve()
new_cache_path = CACHE_PATH / 'cache'



def three_dots(loops, size, threads):
    yaml_path = new_cache_path / 'hub/local/numpy_benchmark/main/numpy_benchmark.yaml'
    yam = load_yaml(yaml_path)
    for x in yam['operators']:
        if x['init_args'] is not None and 'size' in x['init_args'].keys():
            x['init_args']['size'] = size
        if 'threads' in x:
            x['threads'] = threads
    with open(yaml_path, 'w', encoding='utf8') as f:
        yaml.dump(yam, f, default_flow_style=False, sort_keys=False)
    y = []
    time1 = time.time()
    pipeline = towhee.pipeline('local/numpy-benchmark')
    for x in range(loops):
        y.append(pipeline(x)[0][0])
    time2 = time.time()
    runtime = time2 - time1
    return runtime
