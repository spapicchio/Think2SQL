FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore
ENV UV_LINK_MODE=copy
ENV UV_PRERELEASE=allow
ENV UV_COMPILE_BYTECODE=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv/
ENV VIRTUAL_ENV=/opt/venv

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    pkg-config \
    libnccl2 \
    libnccl-dev \
    tmux \
    --allow-change-held-packages \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.6.0 /uv /uvx /bin/

WORKDIR /opt

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=src/,target=src/ \
    uv sync --frozen --no-install-package flash-attn && \
    uv sync --frozen --no-build-isolation