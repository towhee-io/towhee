#!/bin/bash
version=0.3.0

conda create -y -n build-towhee
conda activate build-towhee
conda install -y conda-build anaconda-client
conda config --add channels 'conda-forge'
conda config --add channels 'pytorch'

# convert to platforms
conda build .
conda upload -u towhee-io ${CONDA_PREFIX}/conda-bld/${arch}/towhee-${version}-py_0.tar.bz2

