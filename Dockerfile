FROM ghcr.io/astral-sh/uv:0.10.8 AS uv

FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="opencode-proxy" \
      org.opencontainers.image.description="FastAPI compatibility proxy for OpenCode and OpenAI-compatible upstreams" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROXY_HOST=0.0.0.0 \
    PROXY_PORT=9526

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src

RUN uv sync --locked --no-dev --no-editable

ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser
USER appuser

EXPOSE 9526

# /readyz, not /healthz: /healthz never touches the upstream, so a container
# whose model server is dead would still report healthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9526/readyz', timeout=4).read()"

CMD ["opencode-proxy"]
