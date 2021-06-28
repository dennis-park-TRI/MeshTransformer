ARG BASE_DOCKER_IMAGE
FROM $BASE_DOCKER_IMAGE

ARG python=3.7
ENV PYTHON_VERSION=${python}

# -----------------------------------
# TRI-specific environment variables.
# -----------------------------------
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}

ARG AWS_DEFAULT_REGION
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}

ARG WANDB_ENTITY
ENV WANDB_ENTITY=${WANDB_ENTITY}

ARG WANDB_API_KEY
ENV WANDB_API_KEY=${WANDB_API_KEY}

# -------------------------
# Install core APT packages.
# -------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
      # essential
      build-essential \
      cmake \
      ffmpeg \
      g++-4.8 \
      git \
      curl \
      docker.io \
      vim \
      wget \
      unzip \
      ca-certificates \
      htop \
      libjpeg-dev \
      libpng-dev \
      libavdevice-dev \
      pkg-config \
      # python
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
      python3-tk \
      python${PYTHON_VERSION}-distutils \
      # opencv
      python3-opencv \
    # set python
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------------------------------
# MPI backend for pytorch distributed training (covers both single- and multi-node training).
# -------------------------------------------------------------------------------------------
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install OpenSSH for MPI to communicate between containers
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# -------------------------
# Install core PIP packages.
# -------------------------
# Upgrade pip.
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Core tools.
RUN pip install -U \
        numpy scipy pandas matplotlib seaborn boto3 requests tenacity tqdm awscli scikit-image \
        wandb mpi4py onnx==1.5.0 onnxruntime coloredlogs pycuda

# Install pytorch 1.7 (CUDA 10.1)
RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Install fvcore and detectron2.
ENV FVCORE_CACHE="/tmp"
RUN pip install -U 'git+https://github.com/facebookresearch/fvcore'
RUN python -m pip install detectron2==0.4 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html

#-------------------------------------------------------
# Copy working directory, and optionally install package.
#-------------------------------------------------------
ARG WORKSPACE
COPY . ${WORKSPACE}
WORKDIR ${WORKSPACE}
# # (Optional)
# RUN python setup.py build develop

ENV PYTHONPATH "${PYTHONPATH}:${WORKSPACE}/metro/"

# -------------------------------------
# Install project-specific PIP packages.
# -------------------------------------
# Required for opendr installation
# https://github.com/akanazawa/hmr/issues/82#issuecomment-557430371
RUN apt-get update && apt-get install -y \
        libosmesa6-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        freeglut3-dev

RUN pip install cython opencv-python opendr

# For panoptic segmentation experiments (and for preparing panoptic dataset)
# RUN pip install git+https://github.com/cocodataset/panopticapi.git
# This branch fix the the bug: safely close multiprocessing pools, therefore prevent memory leaking.
RUN pip install git+https://github.com/dennis-park-TRI/panopticapi.git

# -----------
# Final steps
# -----------
WORKDIR ${WORKSPACE}

# # For eGPU on Lenovo P52
# ENV CUDA_VISIBLE_DEVICES=0
