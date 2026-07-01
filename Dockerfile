FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="opencode-proxy" \
      org.opencontainers.image.description="FastAPI compatibility proxy for OpenCode and OpenAI-compatible upstreams" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROXY_HOST=0.0.0.0 \
    PROXY_PORT=9526

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser
USER appuser

EXPOSE 9526

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9526/healthz', timeout=2).read()"

CMD ["opencode-proxy"]
