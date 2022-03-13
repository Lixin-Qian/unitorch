FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
LABEL maintainer="fuliucansheng"
LABEL repository="unitorch"

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   locales \
                   ca-certificates \
                   zip \
                   unzip \
                   libopenmpi-dev \
                   libgl1-mesa-glx \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    mkl \
    torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ENV PATH "$PATH:~/.local/bin"

WORKDIR /workspace

CMD ["/bin/bash"]
