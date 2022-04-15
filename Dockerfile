FROM nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gosu \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    python3 \
    python3-dev \
    python3-pip \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install pipenv

WORKDIR /app

COPY Pipfile Pipfile.lock entrypoint.sh /app/

ENTRYPOINT ["/app/entrypoint.sh"]
