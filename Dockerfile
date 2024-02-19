### IMPORTANT, THIS IMAGE CAN ONLY BE RUN IN LINUX DOCKER
### You will run into a segfault in mac
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_ARGS="-DLLAMA_CUBLAS=ON" \
    FORCE_CMAKE=1 \
    TZ=Europe/Minsk

RUN apt update -y && apt upgrade -y && apt install libreoffice -y && apt install pip -y  \
    && apt install nvidia-driver-535 -y

RUN apt-get update -y

ARG PYVER="3.11"
# Install Python (software-properties-common), Git, and Python utilities
# Learn about the deadsnakes Personal Package Archives, hosted by Ubuntu:
# https://www.youtube.com/watch?v=Xe40amojaXE
RUN apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python$PYVER \
    python3-pip \
    git-all

# Upgrade packages to the latest version
RUN apt-get -y upgrade

# Update PIP (Python's package manager)
RUN python3 -m pip install --upgrade pip
RUN python3 -V
# Set PYVER as the default Python interpreter
RUN update-alternatives --install /usr/bin/python3 python /usr/bin/python$PYVER 1
RUN update-alternatives --set python /usr/bin/python$PYVER
RUN update-alternatives --set python /usr/bin/python$PYVER
RUN python3 -V
# Install poetry
RUN pip install pipx
RUN python3 -m pipx ensurepath
RUN pip install poetry
ENV PATH="/root/.local/bin:$PATH"

# Dependencies to build llama-cpp
RUN apt update && apt install -y \
  libopenblas-dev\
  ninja-build\
  build-essential\
  pkg-config\
  wget

# https://python-poetry.org/docs/configuration/#virtualenvsin-project
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

FROM base as dependencies
WORKDIR /home/worker/app

COPY pyproject.toml poetry.lock ./
#RUN export CMAKE_ARGS="-DLLAMA_CUBLAS=on" && export FORCE_CMAKE=1

RUN poetry install --with local
RUN poetry install --with ui

RUN python3 -m spacy download ru_core_news_md


FROM base as app

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
EXPOSE 8080

WORKDIR /home/worker/app

RUN mkdir local_data
RUN mkdir models

COPY --from=dependencies /home/worker/app/.venv/ .venv
COPY private_gpt/ private_gpt
COPY scripts/setup setup
COPY fern/ fern
COPY *.yaml *.md ./
COPY pyproject.toml poetry.lock ./

ENTRYPOINT .venv/bin/python3 -m private_gpt