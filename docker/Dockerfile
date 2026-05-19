# OngoVoice-RT edge image.
# Targets Raspberry Pi 5 (linux/arm64) by default. Build with:
#   docker buildx build --platform linux/arm64 -t ongovoice/edge:latest .
#
# For x86 dev:
#   docker build -t ongovoice/edge:dev .

FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

# We deliberately don't install whisper / piper here — they need cached
# weights mounted in at runtime. The bare image runs the pipeline with
# the mock providers, which is enough for the dashboard demo.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -e ".[anthropic,openai]"

EXPOSE 8001

HEALTHCHECK --interval=20s --timeout=5s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8001/healthz || exit 1

CMD ["python", "-m", "uvicorn", "ongovoice.api:app", "--host", "0.0.0.0", "--port", "8001"]
