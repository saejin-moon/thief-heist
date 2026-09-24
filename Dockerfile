# Multi-stage Dockerfile for THIEF-HEIST

# Build Rust extension and Python wheel
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY Cargo.toml pyproject.toml README.md ./
COPY src/rs/ ./src/rs/
COPY src/py/ ./src/py/

RUN uv pip install --system maturin
RUN maturin build --release --out dist

# Runtime environment with CUDA PyTorch
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app/src/py"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    fonts-liberation \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/dist/*.whl /tmp/
RUN uv pip install --system --no-cache \
    polars \
    pyarrow \
    matplotlib \
    gymnasium \
    pettingzoo \
    lbforaging \
    pillow \
    scipy \
    pytest \
    && uv pip install --system --no-deps --no-cache /tmp/*.whl \
    && rm -rf /tmp/*.whl

COPY . /app

RUN chmod +x /app/script.sh /app/script_lbf.sh /app/paper/scripts/build.sh

CMD ["python", "src/py/curriculum.py", "--algos", "all", "--concurrent-algos", "2", "--seeds", "0-9", "--rust", "--eval", "--ablations"]
