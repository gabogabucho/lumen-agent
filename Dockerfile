FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal runtime dependencies for TLS/certs and basic health tooling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY lumen /app/lumen

RUN pip install --no-cache-dir .

EXPOSE 3000

# Persist Lumen instance data in /root/.lumen (mount as volume in compose).
CMD ["lumen", "server", "--host", "0.0.0.0", "--port", "3000"]
