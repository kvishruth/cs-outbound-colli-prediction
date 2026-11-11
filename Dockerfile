FROM python:3.13-slim

# LightGBM and XGBoost need the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1

WORKDIR /app

# the libraries first: this layer is only rebuilt when pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# the project is installed in editable mode, so config.py finds /app/data, /app/artifacts and /app/outputs
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# data, models and outputs live outside the image: mount them (see README)
RUN useradd --uid 1000 --create-home app \
    && mkdir -p data/input artifacts/models outputs \
    && chown -R app /app/data /app/artifacts /app/outputs
USER app

ENTRYPOINT ["cs-outbound-colli-prediction"]
CMD ["--help"]
