FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
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

# Python dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade numpy open3d

# Create a directory for building
WORKDIR /opt/teaserpp

# Copy the local repository into the build context
COPY TEASER-plusplus /opt/teaserpp

# Build TEASER++
RUN mkdir -p build && cd build && \
    cmake -DTEASERPP_PYTHON_VERSION=3 .. && \
    make teaserpp_python

# Install Python bindings into the image's Python environment
RUN cd build/python && \
    python3 -m pip install .
