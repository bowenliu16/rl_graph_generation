ARG base_img_with_tag=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
FROM ${base_img_with_tag} as base
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get -y update && apt-get -y install \
    libxrender1 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    apt-get -y clean

RUN conda create -n my-rdkit-env python=3.7
RUN conda init
RUN mkdir -p /usr/src/app/rl_graph_generation
WORKDIR /usr/src/app/rl_graph_generation
