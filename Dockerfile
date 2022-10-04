FROM nvidia/cuda:11.6.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

ENV CUDA_INSTALL_PATH=/usr/local/cuda/
ENV CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc

WORKDIR /build
RUN apt-get update -y && \
 apt-get install -y software-properties-common && \
 add-apt-repository ppa:deadsnakes/ppa && \
 apt-get update -y

RUN apt-get install -y git build-essential libssl-dev wget curl python3.9 python3-pip python3.9-distutils python3.9-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
 update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 && \
 update-alternatives --config python3

RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu116


WORKDIR /syncback
WORKDIR /nucle-ai

COPY ./setup.py ./setup.py
COPY ./setup.cfg ./setup.cfg
COPY ./requirements.txt ./requirements.txt
COPY ./README.md ./README.md
COPY ./requirements-benchmark.txt ./requirements-benchmark.txt
COPY ./src/__init__.py ./src/__init__.py
COPY ./src/nucle/__init__.py ./src/nucle/__init__.py


ENV SKIP_CUDA_ASSERT=1
RUN pip install -e "."
COPY ./ ./