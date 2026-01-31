# Pramana Development Dockerfile
# This is a research project - output is a model and dataset on HuggingFace, not a running service.

FROM nvcr.io/nvidia/pytorch:24.09-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /workspace/pramana

# Copy project files for dependency installation
COPY pyproject.toml README.md ./

# Create minimal package structure for uv to install
RUN mkdir -p src/pramana && \
    echo '"""Pramana package."""\n__version__ = "0.1.0"' > src/pramana/__init__.py

# Install dependencies (will be cached unless pyproject.toml changes)
RUN uv sync --dev

# Copy the rest of the source code
COPY . .

# Reinstall with full source
RUN uv sync --dev

# Default command
CMD ["/bin/bash"]
