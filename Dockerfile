# Start with NVIDIA's latest CUDA base image
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    # python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.10 get-pip.py

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Set the working directory
WORKDIR /app

# RUN python3.10 -m pip install networkx==2.8.8

RUN python3.10 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN python3.10 -m pip install xformers

# Copy requirements file
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV HYDRA_FULL_ERROR 1
