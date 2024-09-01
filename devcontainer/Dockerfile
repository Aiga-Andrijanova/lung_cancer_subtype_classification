FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as base
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

FROM base as build
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \ 
    && apt-get update -q \
    && apt-get install -q -y --no-install-recommends \
        git \
        wget \
        unzip \
        python3-pip

FROM build as runtime

# Python dependencies
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# COPY requirements.txt /tmp/
# RUN --mount=target=/root/.cache/pip,type=cache \
#     pip install --upgrade pip \
#     pip install --ignore-installed -r /tmp/requirements.txt \
#     && rm /tmp/requirements.txt

# ENV PATH=$PATH:/home/vscode/.local/bin
# RUN --mount=target=/home/vscode/.cache/pip,type=cache,sharing=locked \
#     sudo chown -R vscode:vscode /home/vscode/.cache/pip && \
#     pip3 install opencv-python jupyterlab numpy tqdm pandas scikit-learn seaborn pyjson5 marshamllow matplotlib mysql-connector-python 


# X server libs, sudo
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \ 
    && apt-get update -q \
    && apt-get install -q -y --no-install-recommends \
        openssh-client \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        sudo \
        libglib2.0-0 \
    && apt-get upgrade -y

# Add non-root user so we do not have to chown everything outside of the container
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
ENV SHELL /bin/bash
ENV PATH=$PATH:/usr/bin/