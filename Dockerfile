FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY drift_monitor/ drift_monitor/
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim

RUN useradd --create-home appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY drift_monitor/ drift_monitor/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/metrics')" || exit 1

CMD ["python", "-m", "drift_monitor.prometheus_exporter", "--port", "8000"]
