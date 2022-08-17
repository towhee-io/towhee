ARG BASE_IMAGE=ubuntu:20.04
ARG PYTHON_VERSION=3.8

# base image, build command: 
# docker build --platform x86_64 --target towhee-base  -t towhee/towhee-base:latest .
FROM ${BASE_IMAGE} as towhee-base
RUN sed -i "s@http://.*archive.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*
ENV PATH /opt/conda/bin:$PATH

# conda image, build command: 
# docker build --platform x86_64 --target towhee-conda  -t towhee/towhee-conda:latest .
FROM towhee-base as towhee-conda
ARG PYTHON_VERSION=3.8
ARG CUDA_VERSION=11.3
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch
COPY requirements.txt .
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython
ENV CONDA_OVERRIDE_CUDA=${CUDA_VERSION}
RUN /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -y python=${PYTHON_VERSION} pytorch torchvision torchaudio torchtext "cudatoolkit=${CUDA_VERSION}" && \
    /opt/conda/bin/conda clean -ya

# ut image, build command: 
# docker build --platform x86_64 --target towhee-ut  -t towhee/towhee-ut:latest .
FROM towhee-conda as towhee-ut
WORKDIR /workspace