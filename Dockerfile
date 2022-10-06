FROM nvcr.io/nvidia/tensorrt:22.09-py3
ENV DEBIAN_FRONTEND=noninteractive

ENV CUDA_INSTALL_PATH=/usr/local/cuda/
ENV CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc

WORKDIR /build
RUN apt-get update -y && \
 apt-get install -y software-properties-common && \
 add-apt-repository ppa:deadsnakes/ppa && \
 apt-get update -y

RUN apt-get install -y git \
    build-essential \
    libssl-dev \
    wget \
    curl \
    python3.9 \
    python3.9-distutils \
    python3.9-venv \
    python3.9-dev && \
    python3.9 -m ensurepip && \
  update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
  update-alternatives --install /usr/bin/python python /usr/bin/python3.9 2 && \
  update-alternatives --set python /usr/bin/python3.9 && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 && \
  update-alternatives --set python3 /usr/bin/python3.9 && \
  pip install --upgrade pip


RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu116


WORKDIR /syncback
WORKDIR /kernl

COPY ./setup.py ./setup.py
COPY ./setup.cfg ./setup.cfg
COPY ./requirements.txt ./requirements.txt
COPY ./README.md ./README.md
COPY ./requirements-benchmark.txt ./requirements-benchmark.txt
COPY ./src/__init__.py ./src/__init__.py
COPY ./src/nucle/__init__.py ./src/nucle/__init__.py


ENV SKIP_CUDA_ASSERT=1
RUN pip install -e ".[benchmark]"
COPY ./ ./
