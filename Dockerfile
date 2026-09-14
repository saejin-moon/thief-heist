# =============================================================================
# Multi-Stage Production Dockerfile for THIEF-HEIST
# =============================================================================

# Stage 1: Build Rust core and Python wheel using maturin and uv
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install system dependencies & Rust toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project specification files
COPY Cargo.toml pyproject.toml README.md ./
COPY src/rs/ ./src/rs/
COPY src/py/ ./src/py/

# Build optimized Python wheel with Maturin
RUN uv pip install --system maturin
RUN maturin build --release --out dist

# =============================================================================
# Stage 2: Final Runtime Environment (PyTorch + CUDA Acceleration + Fallback)
# =============================================================================
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app/src/py"

# Install uv in runtime container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install the built wheel and scientific dependencies
COPY --from=builder /build/dist/*.whl /tmp/
RUN uv pip install --system --no-cache \
    /tmp/*.whl \
    polars \
    pyarrow \
    matplotlib \
    gymnasium \
    pytest

# Copy application source and scripts
COPY . /app

# Ensure entry script is executable
RUN chmod +x /app/script.sh

# Default entrypoint runs the full curriculum runner
ENTRYPOINT ["python", "src/py/curriculum.py"]
CMD ["--algos", "all", "--concurrent-algos", "2", "--seeds", "0-9", "--rust", "--eval", "--ablations"]
