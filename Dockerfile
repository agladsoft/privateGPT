### IMPORTANT, THIS IMAGE CAN ONLY BE RUN IN LINUX DOCKER
### You will run into a segfault in mac
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_ARGS="-DLLAMA_CUBLAS=ON" \
    FORCE_CMAKE=1

RUN apt update -y && apt upgrade -y && apt install libreoffice -y && apt install pip -y  \
    && apt install nvidia-driver-535 -y

RUN pt-get install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y && apt install python3.11 -y && apt install python3.11-venv -y
RUN apt-get update -y
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.11 1

# Install poetry
RUN pip install pipx
RUN python -m pipx ensurepath
RUN pipx install poetry
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

RUN poetry run python setup

ENTRYPOINT .venv/bin/python -m private_gpt