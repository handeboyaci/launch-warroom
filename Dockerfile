FROM python:3.11-slim

WORKDIR /app

# Install system deps for building wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Cache dependency layer: copy only pyproject.toml first, install
# deps in non-editable mode (editable needs src/ which isn't here
# yet). This layer is cached unless pyproject.toml changes.
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[dev]" 2>/dev/null || true

# Copy source and install package in editable mode.
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

CMD ["pytest", "-v"]
