# Base image with Python 3.9 and Linux
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH \
    MUJOCO_GL=osmesa

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    cmake \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6-dev \
    libglfw3-dev \
    patchelf && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x LTS (required for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create and activate robomimic conda environment with Python 3.9
RUN /opt/conda/bin/conda create -n robomimic_venv python=3.9 -y

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Install Jupyter and all dependencies in the conda environment
RUN /opt/conda/bin/conda run -n robomimic_venv pip install \
    jupyter \
    ipykernel \
    notebook \
    ipywidgets \
    nbconvert \
    jupyterlab

# Register the kernel
RUN /opt/conda/bin/conda run -n robomimic_venv python -m ipykernel install --user --name robomimic_venv --display-name "Python (robomimic_venv)"

# Initialize conda for bash
RUN /opt/conda/bin/conda init bash

# Install PyTorch and torchvision with CPU fallback
RUN /opt/conda/bin/conda run -n robomimic_venv conda install -y pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia || \
    /opt/conda/bin/conda run -n robomimic_venv pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# Install robosuite
# WORKDIR /opt
# RUN git clone https://github.com/ARISE-Initiative/robosuite.git && \
#     cd robosuite && \
#     /opt/conda/bin/conda run -n robomimic_venv pip install -r requirements.txt

RUN /opt/conda/bin/conda run -n robomimic_venv pip install robosuite

# Install additional packages
RUN /opt/conda/bin/conda run -n robomimic_venv pip install wandb

# Set the working directory
WORKDIR /workspace

# Activate Conda environment and start bash when container starts
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && bash"]