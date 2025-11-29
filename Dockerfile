FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    libeigen3-dev \
    libboost-all-dev \
    ca-certificates \
    libsm6 \
    libxext6 \
    libegl1 \
    libgl1 \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Set up environment
RUN git clone --recursive https://github.com/Abuynits/cse291_proj.git
WORKDIR /cse291_proj

# From setup.sh
SHELL ["/bin/bash", "-c"]
# setup uv environment (should be enough for Wan and SAM2 steps)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv sync
ENV PATH="/cse291_proj/.venv/bin/activate:${PATH}"

# get TraceAnything checkpoint
RUN mkdir -p TraceAnything/checkpoints && \ 
    curl -L -o TraceAnything/checkpoints/trace_anything.pt https://huggingface.co/depth-anything/trace-anything/resolve/main/trace_anything.pt?download=true

# place our registration script into the correct directory
RUN mv register_pointclouds.py third_party/DiffusionReg

# Build TEASER++
WORKDIR /cse291_proj/third_party/TEASER-plusplus
RUN mkdir -p build && cd build && \
    cmake -DTEASERPP_PYTHON_VERSION=3 .. && \
    make teaserpp_python

# Install Python bindings into the image's Python environment
RUN cd build/python && \
    python3 -m pip install .

WORKDIR /cse291_proj