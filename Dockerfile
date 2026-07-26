FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN uv pip install --system --no-cache .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health') or exit(1)"

CMD ["uvicorn", "ruitong.main:app", "--host", "0.0.0.0", "--port", "8000"]