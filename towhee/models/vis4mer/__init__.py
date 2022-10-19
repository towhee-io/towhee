# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

try:
    import opt_einsum
except ModuleNotFoundError:
    os.system("pip install opt_einsum")

try:
    import einops
except ModuleNotFoundError:
    os.system("pip install einops")

try:  # Try CUDA extension
    from extensions.cauchy.cauchy import cauchy_mult
    has_cauchy_extension = True
except:
    log.warn(
        "CUDA extension for cauchy multiplication not found. Install by going to extensions/cauchy/ "
        "and running `python setup.py install`. This should speed up end-to-end training by 10-50%"
    )
    has_cauchy_extension = False

try:  # Try pykeops
    import pykeops
    from pykeops.torch import Genred
except ImportError:
    os.system("pip install pykeops")
    import pykeops
    from pykeops.torch import Genred
