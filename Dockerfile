FROM nvidia/cuda:12.0.0-devel-ubuntu22.04
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
    python3.9-dev \
    nano

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
  update-alternatives --install /usr/bin/python python /usr/bin/python3.9 2 && \
  update-alternatives --set python /usr/bin/python3.9 && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 && \
  update-alternatives --set python3 /usr/bin/python3.9

RUN python3.9 -m ensurepip --default-pip --upgrade && \
    pip install --upgrade pip

RUN pip install --pre torch==2.0.0

RUN mkdir /syncback
WORKDIR /kernl

COPY ./setup.py ./setup.py
COPY ./setup.cfg ./setup.cfg
COPY ./requirements.txt ./requirements.txt
COPY ./README.md ./README.md
COPY ./src/kernl/__init__.py ./src/kernl/__init__.py


RUN pip install -e .
COPY ./ ./
