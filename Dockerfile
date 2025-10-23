# Base image with Python 3.9 and Linux
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH \
    MUJOCO_GL=osmesa

# Install system dependencies
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
    patchelf \
    lsb-release \
    gnupg2 \
    software-properties-common \
    mesa-utils \
    x11-xserver-utils && \
    rm -rf /var/lib/apt/lists/*

# Setup ROS Noetic repository
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# Install ROS Noetic, Gazebo, libfranka and franka_ros (official binary packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    ros-noetic-libfranka \
    ros-noetic-franka-ros \
    python3-rosdep \
    python3-catkin-tools \
    python3-wstool && \
    rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init && rosdep update

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

# Disable conda auto-activation to prevent PATH conflicts with ROS
RUN /opt/conda/bin/conda config --set auto_activate_base false

# Install PyTorch and torchvision with CPU fallback
RUN /opt/conda/bin/conda run -n robomimic_venv conda install -y pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia || \
    /opt/conda/bin/conda run -n robomimic_venv pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# Install robosuite
RUN /opt/conda/bin/conda run -n robomimic_venv pip install robosuite

# Install additional packages
RUN /opt/conda/bin/conda run -n robomimic_venv pip install wandb rospkg pytransform3d hidapi

# Install additional packages
RUN /opt/conda/bin/conda run -n robomimic_venv pip install "numpy<2.0" "opencv-python<4.8"

# Configure Gazebo to use GPU
RUN mkdir -p /root/.gazebo && \
    echo "export LIBGL_ALWAYS_SOFTWARE=0" >> /root/.bashrc && \
    echo "export GAZEBO_MODEL_PATH=/usr/share/gazebo-11/models:\$GAZEBO_MODEL_PATH" >> /root/.bashrc

# Setup bashrc to always use conda with ROS sourced first
RUN echo "# ROS setup" >> /root/.bashrc && \
    echo "export PATH=\"/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"" >> /root/.bashrc && \
    echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "# Conda setup" >> /root/.bashrc && \
    echo "export PATH=\"/opt/conda/bin:\$PATH\"" >> /root/.bashrc && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate robomimic_venv" >> /root/.bashrc

# Set the working directory
WORKDIR /rosmimic

# Start with ROS environment by default
CMD ["/bin/bash"]