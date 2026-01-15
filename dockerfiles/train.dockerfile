FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY LICENSE LICENSE
COPY README.md README.md
COPY tasks.py tasks.py
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN mkdir -p models reports/figures

# Install dependencies with caching to speed up subsequent builds
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-install-project

ENTRYPOINT ["sh", "-c", "uv run invoke preprocess-data && uv run invoke train"]
